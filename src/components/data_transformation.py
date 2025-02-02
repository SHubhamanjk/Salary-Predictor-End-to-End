import sys
import os
import pandas as pd
import numpy as np
import pickle
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    processed_train_file_path = os.path.join("artifacts", "processed_train.csv")
    processed_test_file_path = os.path.join("artifacts", "processed_test.csv")
    target_train_file_path = os.path.join("artifacts", "target_train.csv")
    target_test_file_path = os.path.join("artifacts", "target_test.csv")
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl") 


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")

            target_column_name = "Salary"

            train_df = train_df.dropna()
            test_df = test_df.dropna()
    

            train_df = train_df.drop(['Gender'], axis=1)
            test_df = test_df.drop(['Gender'], axis=1)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Starting encoding process...")

            # Label Encoding for Education Level
            le = LabelEncoder()
            input_feature_train_df['Education Level'] = le.fit_transform(input_feature_train_df['Education Level'])
            input_feature_test_df['Education Level'] = le.transform(input_feature_test_df['Education Level'])

            # One-Hot Encoding for Job Title
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            job_title_encoded_train = one_hot_encoder.fit_transform(input_feature_train_df[['Job Title']])
            job_title_encoded_test = one_hot_encoder.transform(input_feature_test_df[['Job Title']])

            df_train_one_hot = pd.DataFrame(job_title_encoded_train, columns=one_hot_encoder.get_feature_names_out(['Job Title']))
            df_test_one_hot = pd.DataFrame(job_title_encoded_test, columns=one_hot_encoder.get_feature_names_out(['Job Title']))

            input_feature_train_df = pd.concat([input_feature_train_df.drop(columns=['Job Title']), df_train_one_hot], axis=1)
            input_feature_test_df = pd.concat([input_feature_test_df.drop(columns=['Job Title']), df_test_one_hot], axis=1)

            logging.info("Encoding completed successfully.")

            os.makedirs("artifacts", exist_ok=True)

            input_feature_train_df=input_feature_train_df[0:260]
            target_feature_train_df=target_feature_train_df[0:260]
            input_feature_test_df=input_feature_test_df[0:111]
            target_feature_test_df=target_feature_test_df[0:111]

            input_feature_train_df=input_feature_train_df.dropna()
            input_feature_test_df=input_feature_test_df.dropna()
            target_feature_train_df=target_feature_train_df.dropna()
            target_feature_test_df=target_feature_test_df.dropna()

            input_feature_train_df.reset_index(drop=True, inplace=True)
            input_feature_test_df.reset_index(drop=True, inplace=True)
            target_feature_train_df.reset_index(drop=True, inplace=True)
            target_feature_test_df.reset_index(drop=True, inplace=True)


            logging.info("Saving processed data...")   
            input_feature_train_df.to_csv(self.data_transformation_config.processed_train_file_path, index=False)
            input_feature_test_df.to_csv(self.data_transformation_config.processed_test_file_path, index=False)
            target_feature_train_df.to_csv(self.data_transformation_config.target_train_file_path, index=False)
            target_feature_test_df.to_csv(self.data_transformation_config.target_test_file_path, index=False)

            logging.info("Processed data saved successfully.")

            preprocessor = {
                "one_hot_encoder": one_hot_encoder,
                "label_encoder": le
            }
            with open(self.data_transformation_config.preprocessor_file_path, 'wb') as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor saved successfully.")

            return (
                self.data_transformation_config.processed_train_file_path,
                self.data_transformation_config.processed_test_file_path,
                self.data_transformation_config.target_train_file_path,
                self.data_transformation_config.target_test_file_path,
                self.data_transformation_config.preprocessor_file_path  
            )

        except Exception as e:
            raise CustomException(e, sys)
