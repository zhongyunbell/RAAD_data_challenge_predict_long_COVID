'''
Feature selection
'''

import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split



def selectFeature(DF, featuretabledir, experimentname = None):
    DF = DF.drop(['patientid'], axis = 1)
    y = DF['has_long_covid_diag']
    X_uncleaned = DF.drop(['has_long_covid_diag'], axis = 1)
    
    X = X_uncleaned.apply(lambda x:x.fillna(x.median()), axis = 0)
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    # test data to be used for testing during classification
    testDF = pd.concat([X_test, y_test], axis = 1)
    
    # training data to find the features
    trainDF = pd.concat([X_train, y_train], axis = 1)
    longSampleDF = trainDF.loc[trainDF['has_long_covid_diag'] == 1]
    shortSampleDF = trainDF.loc[trainDF['has_long_covid_diag'] == 0]
    
    randomShortSampleDF = shortSampleDF.sample(n = len(longSampleDF))
    underDF = pd.concat([longSampleDF, randomShortSampleDF])
    DF = underDF
    
    y = DF['has_long_covid_diag']
    X = DF.drop(['has_long_covid_diag'], axis = 1)
    
    features = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    # random forest feature selection
    rf = RandomForestClassifier(max_depth = 5, 
                                 n_estimators = 50,  
                                 min_samples_split = 10,
                                 max_features = 'log2',
                                 class_weight = 'balanced_subsample',
                                 random_state = 42)
    rf.fit(X_train,y_train)
    
    rf_f_importances = list(zip(features,rf.feature_importances_))
    rf_f_importances.sort(key = lambda x : x[1], reverse = True)
    selected_rf_feature_names = [x[0] for x in rf_f_importances]
    selected_rf_feature_score = [x[1] for x in rf_f_importances]

    # selected_features = [x[0] for x in rf_f_importances if x[1] > 0.005]
    selected_features = selected_rf_feature_names[0:20]
    if 'earliest_to_covid_diag' not in selected_features:
        selected_features.append('earliest_to_covid_diag')
    if 'latest_to_covid_diag' not in selected_features:
        selected_features.append('latest_to_covid_diag')
    
    # write selected features in file
    f = open(os.path.join(featuretabledir, 'top_features.txt'), 'w')
    for feat in selected_features:
        f.write(feat + '\n')
    f.close()
    
    X_new = X[selected_features]
    
    selected_features.append('has_long_covid_diag')     # adding the target label column
    
    
    featureDF = pd.concat([X_new, y], axis = 1)
    
    return featureDF, testDF[selected_features], selected_features

    