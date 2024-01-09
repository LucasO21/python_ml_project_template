# ==============================================================================
# DATA TRANSFORMATION ----
# ==============================================================================

# IMPORTS ----
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object





# DATA TRANSFORMATION CLASS ----
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path \
        .join('artifacts', 'preprocessor_obj.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education',
                                    'lunch', 'test_preparation_course']

            # Numerical Pipeline
            numeric_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            # Categorical Pipeline
            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            # Logging
            logging.info(f'Numerical pipeline created for: {numeric_features}')
            logging.info(f'Categorical pipeline created for : {categorical_features}')

            # Processing Pipeline
            preprocessor = ColumnTransformer(
                transformers = [
                    ('numerical_pipeline', numeric_pipeline, numeric_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            logging.info("Preprocessor object: completed!")

            # Return
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # Data Transformation
    def get_data_transformation(self, train_path, test_path):
        try:
            # Read Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Logging
            logging.info("Data reading: completed!")

            logging.info("Data preprocessing: started!")

            # Preprocessor Object
            #preprocessor = self.get_data_transformer_object()

            # Call Preprocessor
            preprocessor_obj = self.get_data_transformer_object()

            # Feature Labels
            target_column_name = 'math_score'
            numeric_features = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object to training and testing data.')

            # Apply Preprocessor
            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            # Logging
            logging.info(f'Saved preprocessing object.')

            # Save Preprocessor Object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj       = preprocessor_obj
            )

            # Return
            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# Test ----
