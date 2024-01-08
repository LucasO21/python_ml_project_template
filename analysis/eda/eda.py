# ==============================================================================
# EDA ----
# ==============================================================================

# IMPORTS ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import sweetviz as sv



# Data Import ----
df_students = pd.read_csv('analysis/data/stud.csv')


df_students.head()
df_students.info()


# Check Missing ----
df_students.isnull().sum()


# Check Duplicates ----
df_students.duplicated().sum()


# Check Unique Values In Each Column ----
df_students.nunique()


# Summary Statistics ----
df_students.describe()

# - No outliers in the data


# Unique Values In Each Column ----
df_students['parental_level_of_education'].unique()
df_students['lunch'].unique()
df_students['test_preparation_course'].unique()


# Adding Columns ----
df_students = df_students\
    .assign(total_score = lambda x: x['math_score'] + x['reading_score'] + x['writing_score'])\
    .assign(average_score = lambda x: x['total_score']/3)


# Histograms/KDE Plots ----
def get_histogram(data, feature, hue = None):

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplot(121)
    sns.histplot(data = data, x = feature, bins = 30, kde = True, color = 'g')
    plt.subplot(122)
    sns.histplot(data = data, x = feature, kde = True, hue= hue)

    plt.show()

get_histogram(df_students, 'total_score', 'gender')

get_histogram(df_students, 'average_score', 'gender')


# Sweetviz ----
report = sv.analyze(df_students)
report.show_html()