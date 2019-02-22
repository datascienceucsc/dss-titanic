# data_preprocessing.py
#
# Titanic - Machine Learning From Disaster
# Data Science Slugs
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" It may be more effective to fill in the missing age values by the titles in people's names"""
def data_preprocessing(dataset):
    # import data
    # dataset = pd.read_csv('data/train.csv')
    X = dataset.iloc[:, 2:13].values
    Y = dataset.iloc[:, 1].values

    # replace missing data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy= "mean", missing_values = np.nan)
    imputer = imputer.fit(X[:,3])
   
    #X = imputer.fit_transform(X[:, 5]) Testing out new code
    X[:,3] = imputer.transform(X[:,3])


# Random Forest Algorithm 

