import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.logger import logging
app = Flask(__name__)
from src.pipeline.predict_pipeline import PredictPipeline
import os

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun,WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from dotenv import load_dotenv
from src.utils import prompt_template


load_dotenv()
chat_history = []

search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
wiki_wrapper = WikipediaAPIWrapper(max_results=1)
search = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)


groq=ChatGroq(api_key=os.environ.get('GROQ_API_KEY'),model="gemma2-9b-it")
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)

agent = initialize_agent(
    tools=[search, wiki],
    llm=groq,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    agent_executor_kwargs={"prompt": prompt_template},
    handle_parsing_errors=True
)


logging.info("Model loaded successfully")   



data = pd.read_csv("notebook/Dataset09-Employee-salary-prediction.csv")
job_titles = data['Job Title'].unique()

@app.route('/')
def index():
    return render_template('index.html', job_titles=job_titles,chat_history=chat_history,message=None)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    education_level = request.form['education_level']
    years_of_experience = int(request.form['experience'])
    job_title = request.form['job_title']

    if education_level == "Bachelors":
        education_level = 0
    elif education_level == "Masters":
        education_level = 1
    elif education_level == "PhD":
        education_level = 2
    
    model=PredictPipeline()
    input_data = np.zeros(pd.read_csv('artifacts/processed_train.csv').shape[1])
    input_data[0] = age
    input_data[1] = education_level
    input_data[2] = years_of_experience

    job_title_col = f"jobtitle_{job_title.replace(' ', '').lower()}"
    if job_title_col in pd.read_csv('artifacts/processed_train.csv').columns:
        job_title_index = list(pd.read_csv('artifacts/processed_train.csv').columns).index(job_title_col)
        input_data[job_title_index] = 1

        logging.info("PreProcessing Done")

    predicted_salary = model.predict([input_data])[0]
    logging.info("Prediction completed")

    return render_template('index.html', prediction=f"${predicted_salary:,.2f}", job_titles=job_titles,chat_history=chat_history, message=None)

@app.route('/chat', methods=["POST"])
def chat():
    global chat_history

    user_message = request.form["msg"]
    chat_history.append({"sender": "user", "message": user_message})

    try:
        bot_response = agent.run(user_message)

        if hasattr(bot_response, "content"):
            bot_response_text = bot_response.content
        else:
            bot_response_text = str(bot_response)

    except Exception as e:
        bot_response_text = f"Error: {str(e)}"

    chat_history.append({"sender": "bot", "message": bot_response_text})

    return bot_response_text




if __name__ == '__main__':
    app.run(debug=True)
    logging.info("App Is Running")
