import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def linearRegression_predict(Xtrain_LR, ytrain_LR):
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    model.fit(Xtrain_LR, ytrain_LR)
    return model


def RandomForestClassifier_predict(Xtrain_DT, ytrain_DT):
    model = RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=8)
    model.fit(Xtrain_DT, ytrain_DT)
    return model


def review_goodOrbad(star):
    if star <= 2.5:
        return "BAD"
    else:
        return "GOOD"


def main():
    Chinese_Restaurants = pd.read_csv('trimmed.csv')
    le = preprocessing.LabelEncoder()

    Chinese_Restaurants['city'] = le.fit_transform(Chinese_Restaurants.city.values)
    Chinese_Restaurants['NoiseLevel'] = le.fit_transform(Chinese_Restaurants.NoiseLevel.values)

    Chinese_Restaurants['delivery'] = le.fit_transform(Chinese_Restaurants.delivery.values)
    Chinese_Restaurants['TakeOut'] = le.fit_transform(Chinese_Restaurants.TakeOut.values)
    Chinese_Restaurants['GoodForKids'] = le.fit_transform(Chinese_Restaurants.GoodForKids.values)
    Chinese_Restaurants['OutdoorSeating'] = le.fit_transform(Chinese_Restaurants.OutdoorSeating.values)
    Chinese_Restaurants['HasTV'] = le.fit_transform(Chinese_Restaurants.HasTV.values)

    X = Chinese_Restaurants[['city','delivery','NoiseLevel','PriceRange','TakeOut',
                            'GoodForKids','OutdoorSeating','HasTV']].values
    
    # use LinearRegression model to predict
    # training/validation reviews are floats between 1.0-5.0
    y = Chinese_Restaurants['stars'].values
    Xtrain_LR, Xvalid_LR, ytrain_LR, yvalid_LR = train_test_split(X, y)
    LR_model = linearRegression_predict(Xtrain_LR, ytrain_LR)
    print("LinearRegression model predict score(treat reviews as numerics): "
            , LR_model.score(Xvalid_LR, yvalid_LR))


    # use RandomForestClassifier model to predict
    # training/validation reviews are encoded into categories
    Chinese_Restaurants['stars_categories'] = le.fit_transform(Chinese_Restaurants.stars.values)
    y = Chinese_Restaurants['stars_categories'].values
    Xtrain_DT, Xvalid_DT, ytrain_DT, yvalid_DT = train_test_split(X, y)
    DT_model = RandomForestClassifier_predict(Xtrain_DT, ytrain_DT)
    print("RandomForestClassifier model predict score(reviews as categories): "
            , DT_model.score(Xvalid_DT, yvalid_DT))

    # use RandomForestClassifier model to predict
    # reformed training/validation reviews to integers
    # reviews are now integers between 1 and 5
    Chinese_Restaurants['stars_1to5'] = Chinese_Restaurants['stars'].astype(int)
    y = Chinese_Restaurants['stars_1to5'].values
    Xtrain_1to5, Xvalid_1to5, ytrain_1to5, yvalid_1to5 = train_test_split(X, y)
    DTmodel_1to5 = RandomForestClassifier_predict(Xtrain_1to5, ytrain_1to5)
    print("RandomForestClassifier model predict score(predict integer reviews): "
            , DTmodel_1to5.score(Xvalid_1to5, yvalid_1to5))


    # use RandomForestClassifier model to predict
    # classified reviews to two categories: good or bad
    Chinese_Restaurants['review_goodOrbad'] = Chinese_Restaurants['stars'].apply(review_goodOrbad)
    y = Chinese_Restaurants['review_goodOrbad'].values
    Xtrain_Bi, Xvalid_Bi, ytrain_Bi, yvalid_Bi = train_test_split(X, y)
    DTmodel_Bi = RandomForestClassifier_predict(Xtrain_Bi, ytrain_Bi)
    print("RandomForestClassifier model predict score(predict good/bad reviews): "
            , DTmodel_Bi.score(Xvalid_Bi, yvalid_Bi))

if __name__ == '__main__':
    main()