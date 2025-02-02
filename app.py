import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from src.logger import logging
app = Flask(__name__)
from src.pipeline.predict_pipeline import PredictPipeline


logging.info("Model loaded successfully")   



data = pd.read_csv("notebook/Dataset09-Employee-salary-prediction.csv")
job_titles = data['Job Title'].unique()

@app.route('/')
def index():
    return render_template('index.html', job_titles=job_titles)

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

    return render_template('index.html', prediction=predicted_salary, job_titles=job_titles)




if __name__ == '__main__':
    app.run(debug=True)
    logging.info("App Is Running")
