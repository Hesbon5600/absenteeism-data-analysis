# -*- coding: utf-8 -*-

# Import the relevant libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle
import os


class CustomScalar(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scalar = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scalar.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scalar.transform(
            X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# Create a class that will be used to predict new data
class AbsenteeismModel:
    def __init__(self, model_file, scalar_file):
        with open(model_file, 'rb') as model_file, open(scalar_file, 'rb') as scalar_file:
            self.reg = pickle.load(model_file)
            self.scalar = pickle.load(scalar_file)
            self.data = None

    # Take the *.csv file and preprocess it
    def load_and_clean_data(self, data_file):
        # import the data
        df = pd.read_csv(data_file, delimiter=',')
        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        # drop the ID column
        df = df.drop(['ID'], axis=1)
        # to preserve the code we've created, we will add a column with NAN strings
        df['Absenteeism Time in Hours'] = 'NaN'

        # create a separate dataframe containing dummy variables for all the available Reasons
        reason_columns = pd.get_dummies(
            df['Reason for Absence'], drop_first=True)

        # split reason_columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        # to avoid multicollinnearity, drop the `Reason for Absence` column from df
        df = df.drop(['Reason for Absence'], axis=1)

        # concatenate df with the 4 Reasons for Absence
        df = pd.concat([df, reason_type_1, reason_type_2,
                        reason_type_3, reason_type_4], axis=1)

        # assign names to the 4 reason types columns
        # Note: There is a more universal version of this code. However, this will best suit our current purpose
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                        'Daily Work Load Average', 'Body Mass Index', 'Education',
                        'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

        df.columns = column_names

        # reorder the columns
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date',
                                  'Transportation Expense', 'Distance to Work', 'Age',
                                  'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # conver the date column into datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # create a list with month values retrieved from the 'Date' column
        list_months = []
        list_months = list(map(lambda x: df['Date'][x].month, list(
            range(df.shape[0]))))

        # insert the values into a new colun in the df called 'Month Value'
        df['Month Value'] = list_months

        # create a new feature called 'Day of the Week'
        def day_of_week(x): return x.weekday()
        df['Day of the Week'] = df['Date'].apply(
            day_of_week)

        # drop the 'Date' column from the df
        df = df.drop('Date', axis=1)

        # reorder the columns
        column_names_updated = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                                'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education',
                                'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_updated]

        # map 'Education' variables; the results is a dummy
        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})

        # replace the NaN values
        df = df.fillna(value=0)

        # drop the original 'Absenteeism Time in Hours'
        df = df.drop(['Absenteeism Time in Hours'], axis=1)

        # drop the variables we decided we will not use
        df = df.drop(
            ['Distance to Work', 'Daily Work Load Average', 'Day of the Week'], axis=1)

        # declare a new variable called processed_data
        self.preprocessed_data = df.copy()

        # scale the data
        self.data = self.scalar.transform(self.preprocessed_data)

    # a function which outputs the probability of a data point to be 1
    def predict_probability(self):
        if self.data is not None:
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred

    # a function that outputs 0 or 1 based on the model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probability and
    # add add columns with these values at the end of the df
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.predict_probability()
            self.preprocessed_data['Prediction'] = self.predicted_output_category(
            )
            return self.preprocessed_data
