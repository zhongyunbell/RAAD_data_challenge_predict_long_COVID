'''
luminous - long covid detection
4 step
    1. Preprocess the data to create feature tables
    2. Merge all the table
    3. Feature selection
    4. Classification
'''

import os
import click
import warnings
warnings.filterwarnings('ignore')

from common import CLICK_SETTINGS
from preprocess_raw_input_files import *
from clean import *
from featureselection import *
from classification import *

@click.command(context_settings=CLICK_SETTINGS)

@click.option("-d", "--datapath", required = True, type = click.Path(exists = True, dir_okay = True), help = 'Train/Test Data directory path')
@click.option("-f", "--featuretabledir", required = True, type = click.Path(dir_okay = True), help = 'Directory path that stores processed data. DO NOT provide an emptry directory path. Directory will be created with given path.')
@click.option("-t", "--experimenttype", required = True, type=click.Choice(['train', 'test'], case_sensitive = False), help = "Train or Test")
@click.option("-n", "--experimentname", required = True, help = "Name of the experiment")


def main(datapath, featuretabledir, experimenttype, experimentname):
    
    # Step 1: pre-processing the data and store in directory "featureTables"
    if experimenttype.lower() == 'train':
        if not os.path.exists(featuretabledir):
            os.mkdir(featuretabledir)
            print('Data directory created')
            processRawData(datapath, featuretabledir, experimenttype)
        else:
            print('Processsed data exists')

        # Step 2: Merge tables from directory "featureTables"
        CleanedDF = cleanTables(featuretabledir, experimenttype)

        # Step 3: Feature selection
        trainDF, testDF, selected_features = selectFeature(CleanedDF, featuretabledir, experimentname)
        trainDF['patientid'] = CleanedDF['patientid']
        testDF['patientid'] = CleanedDF['patientid']
        
        # Step 4: Classification
        classify(experimenttype, featuretabledir, testDF, trainDF)
        
    if experimenttype.lower() == 'test':
        # Step 1: Pre-process teh data and store in directory
        if not os.path.exists(os.path.join(featuretabledir, experimenttype + "_master_table_outer_merge.txt")):
            processRawData(datapath, featuretabledir, experimenttype)
        else:
            print('Processed test data exists')
        
        # Step 2: 
        CleanedDF = cleanTables(featuretabledir, experimenttype)
        
        # Step 3:
        # read the top features and format data with only selected features
        try:
            f = open(os.path.join(featuretabledir, 'top_features.txt'), 'r')
            features = list()
            for line in f.readlines():
                features.append(line.strip())
            f.close()
        except:
            print('Top Feature file not found')
        
        TestDF = CleanedDF[features]
        TestDF['patientid'] = CleanedDF['patientid']
        
        
        # Step 4: Classification
        classify(experimenttype, featuretabledir, TestDF)
        
            
    
if __name__ == '__main__':
    main()
