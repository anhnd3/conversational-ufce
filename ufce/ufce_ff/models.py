import pandas as pd
import numpy as np
import urllib, os
import dice_ml
from dice_ml.utils import helpers
from alibi.explainers import CEM
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
#import recourse as rs

class Model:
    def __init__(self, path, datafile="Bank_Loan_processed_data.csv"):
        self.datafile = datafile
        self.path = path
        self.df = pd.read_csv(datafile)
        del self.df['Unnamed: 0']
        self.linear_reg = LinearRegression()
        self.random_forest = RandomForestRegressor()
        self.svm = sklearn.svm.SVC(kernel='rbf', probability=True)

    def split(self, test_size):
        X = np.array(self.df[['Age', 'Experience', 'Income', 'Family', 'Education', 'Mortgage', 'Securities Account',
                              'CD Account', \
                              'Online', 'CreditCard']])
        # X = np.array(self.df.loc[ : , df.columns != 'Personal Loan'])
        y = np.array(self.df['Personal Loan'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)

    def fit(self):
        self.model = self.random_forest.fit(self.X_train, self.y_train)

    def predict(self, input_value):
        if input_value == None:
            result = self.random_forest.fit(self.X_test)
        else:
            result = self.random_forest.fit(np.array([input_value]))
        return result
