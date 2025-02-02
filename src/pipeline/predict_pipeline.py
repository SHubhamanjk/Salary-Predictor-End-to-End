import sys
from src.exception import CustomException
from src.utils import load_object
import os
from src.logger import logging


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)
            logging.info("Model loaded successfully")
            preds=model.predict(features)
            logging.info("Prediction completed")

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


