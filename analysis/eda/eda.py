# ==============================================================================
# EDA ----
# ==============================================================================

# IMPORTS ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *



# Data Import ----
df_students = pd.read_csv('analysis/data/stud.csv')

df_students.info()