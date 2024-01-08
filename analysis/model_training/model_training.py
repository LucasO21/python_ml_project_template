# ==============================================================================
# MODEL TRAINING ----
# ==============================================================================

# Imports ----------------------------------------------------------------------

# Basic Import ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Modelling ----
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Other ----
import warnings


# Data Import ----
df_students = pd.read_csv('analysis/data/stud.csv') \
    .assign(total_score=lambda x: x['math_score'] + x['reading_score'] + x['writing_score']) \
    .assign(average_score=lambda x: x['total_score']/3)

# X and y ----
X = df_students.drop(['math_score'], axis=1).columns.tolist()
y = 'math_score'

# Data Types ----
num_features = df_students.drop(['math_score'], axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df_students.drop(['math_score'], axis=1).select_dtypes(include=['object']).columns.tolist()

# Column Transformer ----
numeric_transformer = StandardScaler()
category_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScaler', numeric_transformer, num_features),
        ('OneHotEncoder', category_transformer, cat_features)
    ]
)

# Fit & Transform (Train Data) ----
X = preprocessor.fit_transform(df_students[X])

# Train Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, df_students[y], test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Models ----
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list =[]

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)


    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)

    print('='*35)
    print('\n')


# Model Comparison ----
model_comparison = pd.DataFrame({'model': model_list, 'r2_score': r2_list}) \
    .sort_values(by='r2_score', ascending=False)
