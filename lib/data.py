import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from enum import Enum

class RiskGroup(Enum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

class DataType(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

class InsuranceData:
    """
    Loads the data from the given path.

    https://colab.research.google.com/github/embarced/notebooks/blob/master/mlops/generate.ipynb?hl=en

    This is a database of customers of an insurance company. Each data point is one customer. Risk is expressed as a number between 0 and 1. 1 meaning highest and 0 meaning lowerst risk of having an accident.
    """

    def __init__(self, dims=2, path='https://raw.githubusercontent.com/djcordhose/ml-resources/master/notebooks/mlops/insurance-customers-risk-1500.csv'):
        self.df = pd.read_csv(path)
        self.dims = dims

    def get_raw_data(self):
        return self.df

    def get_correlations(self, just_X=True):
        cm = self.df.corr()
        if just_X:
            cm = cm.iloc[:3, :3]
        return cm

    def get_X(self):
        if self.dims == 2:
            X = self.df.drop(['risk', 'group', 'miles'], axis='columns').values
            # reorder, first age, then speed to match plotting
            X = pd.DataFrame(np.array([X[:, 1], X[:, 0]]).T)
        else:
            X = self.df.drop(['risk', 'group'], axis='columns')
        return X.values

    def get_y(self):
        y = self.df['group']
        return y.values

    def get_data(self):
        X = self.get_X()
        y = self.get_y()
        return X, y

    def get_split(self, test_size=0.2):
        X = self.get_X()
        y = self.get_y()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        return X_train, X_val, y_train, y_val