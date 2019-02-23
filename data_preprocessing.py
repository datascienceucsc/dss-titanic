# data_preprocessing.py
#
# Titanic - Machine Learning From Disaster
# Data Science Slugs
#

import numpy as np
import pandas as pd

# import data
dataset = pd.read_csv('data/train.csv')
X = dataset.iloc[:, 2:13].values
Y = dataset.iloc[:, 1].values

# replace missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy= "mean", missing_values = np.nan)
X[:, 3:4] = imputer.fit_transform(X[:, 3:4])