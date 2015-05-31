"""wnvutils.py

"""

__author__ = 'chrisk'

import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

#
# Datasets
#
# Train
# Date, Address, Species, Block, Street, Trap, AddressNumberAndStreet
# Latitude, Longitude, AddressAccuracy, NumMosquitos, WnvPresent
#
# CULEX PIPIENS/RESTUANS    4752
# CULEX RESTUANS            2740
# CULEX PIPIENS             2699
# CULEX TERRITANS            222
# CULEX SALINARIUS            86
# CULEX TARSALIS               6
# CULEX ERRATICUS              1
#
# 138 addresses
#

#
# Starting from script
# https://www.kaggle.com/abhishek/predict-west-nile-virus/vote-me-up
#

def load_datasets(inputdir):
    """Read the data files into a dict, keyed on filename. """

    datasets = {}
    files = os.listdir(inputdir)

    for f in [f for f in files if f.endswith('.csv')]:
        # sampleSubmission doesn't have a date column
        datecol = ['Date'] if f != 'sampleSubmission.csv' else None
        datasets[f] = pd.read_csv(os.path.join(inputdir, f),
                                  parse_dates=datecol)

    return datasets

def clean_weather(weather):
    """"""

    weather.replace("M", float("NaN"), inplace=True)
    weather.replace("-", float("NaN"), inplace=True)
    weather.replace("T", float("NaN"), inplace=True)
    weather.replace(" T", float("NaN"), inplace=True)
    weather.replace("  T", float("NaN"), inplace=True)
    weather.drop("CodeSum", axis=1, inplace=True)

    weather1 = weather[weather["Station"] == 1]
    weather2 = weather[weather["Station"] == 2]

    rows, rows1, rows2 = (weather.shape[0],
                          weather1.shape[0],
                          weather2.shape[0])

    weather = pd.merge(weather1, weather2, on="Date")
    weather.drop(["Station_x", "Station_y"], axis=1, inplace=True)

    newrows = weather.shape[0]
    # sanity check the rows
    assert(rows1 + rows2 == rows)
    assert(rows1 == newrows)

    return weather

def clean_train_test(train):
    """Clean up the test / training data. """

    train["Month"] = train.Date.apply(lambda x: x.month)
    train["Year"] = train.Date.apply(lambda x: x.year)
    train["Day"] = train.Date.apply(lambda x: x.day)

    # Doesn't actually seem to help
    #train["Latitude_int"] = train.Latitude.apply(int)
    #train["Longitude_int"] = train.Longitude.apply(int)

    c2d = ["Id", "Address", "AddressNumberAndStreet", "WnvPresent",
           "NumMosquitos"]

    for column in c2d:
        if column in train.columns:
            train.drop(column, axis=1, inplace=True)

    return train

def clean_train_test2(train, test):
    """Perform operations requiring both train and test. """

    # Species, Street, Trap
    labeller = LabelEncoder()
    labeller.fit(np.concatenate((train.Species.values, test.Species.values)))
    train.Species = labeller.transform(train.Species.values)
    test.Species = labeller.transform(test.Species.values)

    labeller.fit(np.concatenate((train.Street.values, test.Street.values)))
    train.Street = labeller.transform(train.Street.values)
    test.Street = labeller.transform(test.Street.values)

    labeller.fit(np.concatenate((train.Trap.values, test.Trap.values)))
    train.Trap = labeller.transform(train.Trap.values)
    test.Trap = labeller.transform(test.Trap.values)

    return train, test

