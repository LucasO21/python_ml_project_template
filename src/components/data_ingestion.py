

# IMPORTS ----
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from exception import CustomException
from logger import logging
from components.data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainer, ModelTrainingConfig

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# DATA INGESTION CLASS ----
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'data', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_data_ingestion(self):
        logging.info('Data Ingestion Started...')
        try:
            df = pd.read_csv('analysis/data/student.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw Data Ingestion Completed...')

            logging.info('Train Test Split Ingestion Initiated...')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train Test Split Ingestion Completed...')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Error in Data Ingestion...")
            raise CustomException(e, sys)

# Test ----
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.get_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array,_ = data_transformation.get_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.get_trained_model(train_array, test_array, DataTransformationConfig.preprocessor_obj_file_path))

