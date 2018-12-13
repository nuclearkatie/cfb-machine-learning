import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_remove_scale(data, test_size=0.2):
    """
    X_train, X_test, y_train, y_test, home_train, away_train, home_test, away_test
      = split_remove_scale(data, test_size=0.2)

    Use only for splitting data into a training and testing set. Use
    remove_scale if splitting not needed
    """

    X, y = assign_Xy(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size)

    X_train, home_train, away_train = remove_teams(X_train)
    X_test, home_test, away_test = remove_teams(X_test)

    X_train = scaler(X_train)
    X_test = scaler(X_test)

    return X_train, X_test, y_train, y_test, home_train, away_train, home_test, away_test


def remove_scale(data):
    """
    X, y, home, away = remove_scale(data)

    Use only when splitting is not needed, i.e. this is a testing set only.
    If splitting is needed, use split_remove_scale
    """

    X, y = assign_Xy(data)
    X, home, away = remove_teams(X)
    X = scaler(X)

    return X, y, home, away

def assign_Xy(data):

    X = data.iloc[:, 0:(len(data.columns)-1)]
    y = data.iloc[:, (len(data.columns)-1)]

    print("Features and labels split successfully")

    return X, y


def remove_teams(X):

    home = X.iloc[:, 0]
    away = X.iloc[:, 1]

    X = X.iloc[:, 2:(len(X.columns)-1)]

    print("Team names removed")

    return X, home, away

def scaler(X):

    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)

    print("Feature vectors scaled")

    return X
