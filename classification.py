'''
Classify the data and calculate the AUC
'''

import os
import numpy as np
import pandas as pd 
import pickle
import pyarrow.csv as pv
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



def classify(experiment, featuretabledir, testDF, trainDF = None):
    
    if experiment.lower() == 'train':
        patientids = list(testDF['patientid'])
        testDF = testDF.drop(['patientid'], axis = 1)
        testLabel = list(testDF['has_long_covid_diag'])
        testData = testDF.drop(['has_long_covid_diag'], axis = 1)

        trainDF = trainDF.drop(['patientid'], axis = 1)
        y = trainDF['has_long_covid_diag']
        X = trainDF.drop(['has_long_covid_diag'], axis = 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

        
        ############### xgBoost ###############
#         # Define the parameter grid for grid search for parameter tuning
#         param_grid = {'booster':['gbtree', 'gblinear'],
#                       'max_depth': [3, 6, 10],
#                       'learning_rate': [0.1, 0.5, 1.0],
#                       'objective': ['reg:squarederror', 'binary:logistic'],
#                       'eval_metric': ['rmse', 'auc'],
#                      }

#         clf = xgb.XGBClassifier()

#         # Perform grid search
#         grid_search = GridSearchCV(clf, param_grid, cv=5)
#         grid_search.fit(X_train, y_train)

#         # Print the best parameters and the best score
#         print(grid_search.best_params_)
#         print(grid_search.best_score_)

        
        clf = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.01, max_depth = 3, alpha = 1, n_estimators = 50)
        

        clf.fit(X_train, y_train)
        pickle.dump(clf, open(os.path.join(featuretabledir, 'trained_model.sav'), 'wb'))   # saving the trained model

        # run model on test data
        y_pred_prob = clf.predict_proba(testData)
        y_pred = clf.predict(testData)
        
        recall = roc_auc_score(testLabel, y_pred)
        print('Recall:',  recall)
        print('Confusion Matrix')
        print(confusion_matrix(testLabel, y_pred))
    
    else:
        patientids = list(testDF['patientid'])
        testDF = testDF.drop(['patientid'], axis = 1)
        loaded_model = pickle.load(open(os.path.join(featuretabledir, 'trained_model.sav'), 'rb'))
        y_pred_prob = loaded_model.predict_proba(testDF)
        
        # write in a parquet file
        filename = 'predictions.csv'
        f = open(filename, 'w')
        f.write('patientid,prediction' + '\n')
        for i in range(len(patientids)):
            f.write(patientids[i] + ',' + str(y_pred_prob[i][0]) + '\n')
        f.close()
        
        # convert to parquet
        table = pv.read_csv(filename)
        pq.write_table(table, filename.replace('csv', 'parquet'))
        