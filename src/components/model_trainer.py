import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try:
            X_train = pd.read_csv(X_train)
            y_train = pd.read_csv(y_train)
            X_test = pd.read_csv(X_test)
            y_test = pd.read_csv(y_test)

            y_train = y_train.to_numpy().ravel()
            y_test = y_test.to_numpy().ravel()


            logging.info("Read train and test data successfully")

            model=KNeighborsRegressor(algorithm='kd_tree', leaf_size=10, metric='manhattan', n_neighbors=20, p=1, weights='distance')
            logging.info("Model training initiated")
            model.fit(X_train,y_train)
            logging.info("Model training completed")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            predicted=model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            print(f"R2 Square: {r2_square*100}")
            logging.info(f"R2 Square: {r2_square*100}")
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)