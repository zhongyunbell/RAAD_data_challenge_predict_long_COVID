'''
Clean all table from the directory path
'''

import os
import pandas as pd 
import numpy as np 
import glob
from functools import reduce

def cleanTables(directoryPath, experiment):
    try:
        DF = pd.read_csv(os.path.join(directoryPath, experiment + "_master_table_outer_merge.txt"), sep = '\t')
    except:
        print('Prorcessed data file missing')
        
    if experiment.lower() == 'train':
        try:
            targetDF = pd.read_csv(os.path.join(directoryPath, "target.txt"), sep = '\t')
        except:
            print('Target file missing')

    if experiment.lower() == 'train':
        mergedDF = pd.merge(DF, targetDF, on = 'patientid')
    
    else:
        mergedDF = DF
    
    allColumns = list(mergedDF.columns)  
    
    # replace the non-numrical values with NaN in the following columns
    mergedDF['Baso_max'] = pd.to_numeric(mergedDF['Baso_max'], errors = 'coerce')
    mergedDF['Lympho_max'] = pd.to_numeric(mergedDF['Lympho_max'], errors = 'coerce')
    mergedDF['Creatinine_max'] = pd.to_numeric(mergedDF['Creatinine_max'], errors = 'coerce')
    mergedDF['WBC_max'] = pd.to_numeric(mergedDF['WBC_max'], errors = 'coerce')
    mergedDF['Glucose_max'] = pd.to_numeric(mergedDF['Glucose_max'], errors = 'coerce')
    mergedDF['CO2_max'] = pd.to_numeric(mergedDF['CO2_max'], errors = 'coerce')
    mergedDF['Mono_max'] = pd.to_numeric(mergedDF['Mono_max'], errors = 'coerce')
    mergedDF['PLT_max'] = pd.to_numeric(mergedDF['PLT_max'], errors = 'coerce')
    mergedDF['Eosin_max'] = pd.to_numeric(mergedDF['Eosin_max'], errors = 'coerce')
    mergedDF['Neutro_max'] = pd.to_numeric(mergedDF['Neutro_max'], errors = 'coerce')
    mergedDF['RBC_max'] = pd.to_numeric(mergedDF['RBC_max'], errors = 'coerce')
 
    # convert the rest of categorical columns to one-hot-encoding
    ethnicity_hot = pd.get_dummies(mergedDF['ethnicity'])
    race_hot = pd.get_dummies(mergedDF['race'])
    alcohol_hot = pd.get_dummies(mergedDF['ALCOHOL'])
    smoke_hot = pd.get_dummies(mergedDF['SMOKE'])
    gender_hot = pd.get_dummies(mergedDF['gender'])
    influenza_hot = pd.get_dummies(mergedDF['Influenza'])
    covid_hot = pd.get_dummies(mergedDF['COVID-19'])
 
    
    # concatenate after converting categorical columns to one-hot-encoded columns
    mergedDF = pd.concat([mergedDF, ethnicity_hot, race_hot, alcohol_hot, smoke_hot, gender_hot, influenza_hot, covid_hot], axis = 1)
    
    # drop the columns with categorical values
    mergedDF = mergedDF.drop(['ethnicity', 'race', 'ALCOHOL', 'SMOKE', 'gender', 'index_month_year', 
                              'birth_yr', 'Influenza', 'COVID-19',
                             ], axis = 1)
    
    allColumns = list(mergedDF.columns)
    
    # only get the strings
    allStringColumns = [x for x in allColumns if isinstance(x, str)]
    
    # remove columns with 'Un' in their name, i.e. Unknown, Unspecified etc.
    substring = 'Un'
    filteredColumns = [x for x in allStringColumns if substring.lower() not in x.lower()]

    # cleanedDF with only good columns
    cleanedDF = mergedDF[filteredColumns]
    
    # check if any categorical column left
    categoricalColumns = cleanedDF.columns[cleanedDF.dtypes == object]
    
    
    return cleanedDF
