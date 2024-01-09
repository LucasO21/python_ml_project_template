# ==============================================================================
# MODEL TRAINING ----
# ==============================================================================

# IMPORTS ----
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report

from exception import CustomException
from logger import logging
from utils import save_object, get_model_evaluation



# MODEL TRAINING CLASS ----
@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'trained_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()


    def get_trained_model(self, train_array, test_array, preprocessor_path):
        try:
            logging.info('Start Data Splitting...')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = get_model_evaluation(models = models,
                                                      X_train = X_train,
                                                      y_train = y_train,
                                                      X_test = X_test,
                                                      y_test = y_test)

            # Get Best Model Score
            best_model_score = max(sorted(model_report.values()))

            # Get Best Model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            threshold = 0.7

            if best_model_score < threshold:
                raise CustomException(f'Best Model Score: {best_model_score} is less than threshold: {threshold}')

            logging.info(f'Best Model: {best_model_name} with score: {best_model_score}')

            save_object(
                file_path = self.model_training_config.trained_model_file_path,
                obj       = best_model
            )

            prediction = best_model.predict(X_test)

            r2_square = r2_score(y_test, prediction)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)