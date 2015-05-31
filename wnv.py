__author__ = 'chrisk'

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score

import wnvutils

def load_data():

    datasets = wnvutils.load_datasets("./input")
    return (datasets["weather.csv"],
            datasets["train.csv"],
            datasets["spray.csv"],
            datasets["test.csv"])


def desc_df(df):
    print(df.shape)
    print(df.columns)

def main():

    weather, train, spray, test = load_data()
    target = train.WnvPresent.values
    idcol = test.Id.values

    weather = wnvutils.clean_weather(weather)

    train = wnvutils.clean_train_test(train)
    test = wnvutils.clean_train_test(test)

    train, test = wnvutils.clean_train_test2(train, test)

    train = train.merge(weather, on="Date")
    test = test.merge(weather, on="Date")

    train.drop("Date", axis=1, inplace=True)
    test.drop("Date", axis=1, inplace=True)

    desc_df(train)

    train = train.ix[:, pd.notnull(train).any(axis=0)]
    test = test.ix[:, pd.notnull(test).any(axis=0)]

    desc_df(train)

    imputer = Imputer()
    traina = imputer.fit_transform(train)
    testa = imputer.fit_transform(test)

    training = np.random.choice([True, False], size=train.shape[0], p=[0.8, 0.2])

    rfc = ensemble.RandomForestClassifier(oob_score=True)
    rfc.fit(traina[training], target[training])
    print("oob score:", rfc.oob_score_)

    for name, imp in sorted(zip(train.columns, rfc.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        print(name, ":", imp)

    predictions = rfc.predict(traina[~training])

    print("Accuracy:", (predictions == target[~training]).mean())

    predictions = rfc.predict_proba(traina[~training])
    np.savetxt("/tmp/predictions.txt", predictions[:,1])
    print(predictions[:,1])

    print("ROC AUC Score:", roc_auc_score(target[~training], predictions[:,1]))

    print(test.shape)

if __name__ == "__main__":
    main()
