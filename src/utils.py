# ==============================================================================
# UTILS ----
# - DB Clinets
# - Common Functions
# ==============================================================================

# Imports ----
import os
import sys
import dill
import numpy as np
import pandas as pd
from exception import CustomException


# Save Path Function ----
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)
