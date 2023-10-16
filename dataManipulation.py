"""
Created on Sat Nov 10 23:20 2022
@author: tylerliddell & aomerCS
Code referenced from: https://www.youtube.com/watch?v=M_Cu7r9gik4
File contains methods necessary for accessing raw and filtered data
"""

import os.path
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


# Returns the base, unedited Dataset
def get_dataframe():
    return pd.read_csv(os.path.join("../data/", "creditcard.csv"))


# Returns the edited Dataset
def wrangled_dataframe():
    # Load the Dataset
    df = get_dataframe()

    # Scales the Data in the Amount column according to IQR (Inter-Quartile Range)
    df["Amount"] = RobustScaler().fit_transform(df["Amount"].to_numpy().reshape(-1, 1))

    # Reformat the Data in the Time column from 0 to 1
    time = df["Time"]
    df["Time"] = (time - time.min()) / (time.max() - time.min())

    # Returns the dataset without the unneeded features
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
    df = df.drop(columns=["V13", "V15", "V22", "V23", "V24", "V25", "V26"])

    # Randomizes all items in Dataset
    df = df.reindex(np.random.permutation(df.index))

    return df


# Taken from https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
# Returns X,y and arrays from df split into test, train and validate using the ratios in the inputs
# e.g. get_X_y_train_test_val(wrangled_dataframe(), 0,75, 0.15, 0.10)
def get_X_y_train_test_val(df, test_ratio: int, train_ratio: int, val_ratio: int):
    # Initialize X and y
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # Split the data Test/Train/Validate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_ratio, train_size=train_ratio, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_ratio / (test_ratio + val_ratio),
        train_size=val_ratio / (test_ratio + val_ratio),
        random_state=42,
    )

    return X, y, X_train, y_train, X_test, y_test, X_val, y_val


def upsample_train_data(X_train, y_train):
    upsampler = RandomOverSampler(random_state=2)
    return upsampler.fit_resample(X_train, y_train)


def downsample_train_data(X_train, y_train):
    downsampler = RandomUnderSampler(random_state=2)
    return downsampler.fit_resample(X_train, y_train)


def smote_train_data(X_train, y_train):
    smote = SMOTE()
    return smote.fit_resample(X_train, y_train)
