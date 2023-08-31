### Clean all data tables, output two master sheets (1) inner join and (2) outer join (for raw test dataframes)
### RAAD challenge, seeing through the fogs, long COVID prediction
### Jan 19th 
### Julie Huang
### Usage: python3 preprocess_raw_input_files_Julie.py -d /challenge/seeing-through-the-fog/data/train_data -o /home/huangz36/test_process


#!pip install fastparquet


import pandas as pd
import numpy as np
import sys
import os as os
import argparse
import math
import time
import functools
from functools import reduce
import re
from dateutil import tz


## helper functions
def write_list_to_txt(output_file, lines):
    with open(output_file, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

## helper functions            
def read_txt_to_list(input_file):
    with open(input_file) as file_in:
        list_of_lines = []
        for line in file_in:
            list_of_lines.append(line.rstrip())
    return list_of_lines


def clean_up_dia_table(df_dia, output_txt_file_location, output_days_to_response_txt):
    '''
    df_dia is the name of the raw pandas dataframe read from the read_parquet
    output_txt_file_location is the output file location and name, eg:/home/huangz36/dia_table_pivot_by_ICD10_code.txt
    '''
    #print(df_dia.shape)
    df_dia_dedup = df_dia.drop_duplicates()

    ## Extract ICD10
    df_dia_ICD10 = df_dia_dedup[df_dia_dedup['diagnosis_cd_type']=='ICD10']
    df_dia_ICD10['dia_icd10_short'] = df_dia_ICD10['diagnosis_cd'].astype(str).str[0]
    df_dia_ICD10_short = df_dia_ICD10[['patientid', 'days_to_covid_diag', 'dia_icd10_short']].drop_duplicates()

    ## Record "present/absent", as 1 or 0 for ICD10_status
    df_dia_ICD10_short_by_patient = df_dia_ICD10_short[['patientid', 'dia_icd10_short']].drop_duplicates()
    df_dia_ICD10_short_by_patient['ICD10_status'] = 1
    df_dia_ICD10_pivot_table = df_dia_ICD10_short_by_patient.pivot(index=["patientid"], columns=["dia_icd10_short"], values=['ICD10_status']).fillna(0).reset_index()
    ## Collapse header
    new_column_names = []
    for (row1, row2) in df_dia_ICD10_pivot_table.columns:
        if row1=='patientid':
            column_name = str(row1)
        else:
            column_name = str(row1) + "_" + str(row2)  
        new_column_names.append(column_name)
    df_dia_ICD10_pivot_table.columns = new_column_names
    #print(df_dia_ICD10_pivot_table.head())
    ## Output to txt file
    # df_dia_ICD10_pivot_table.to_csv(output_txt_file_location, sep='\t', index=False, header=True)

    df_dia_earliest_latest = df_dia_dedup.groupby(['patientid'])[['days_to_covid_diag']].agg(['min','max']).reset_index().drop_duplicates()
    df_dia_earliest_latest.columns = ['patientid', 'earliest_dia_response_to_covid_diag', 'latest_dia_response_to_covid_diag']
    # df_dia_earliest_latest.to_csv(output_days_to_response_txt, sep='\t', index=False, header=True) 
    return (df_dia_earliest_latest, df_dia_ICD10_pivot_table)


def clean_up_med_table(df_med, days_to_response_output_txt_file_location, df_top_100_drug, drug_by_category_output_txt_file_location):
    '''
    df_med: the name of the raw pandas dataframe read from the read_parquet
    output_txt_file_location: the output file location and name, eg:/home/huangz36/med_table.txt
    df_top_100_drug: the preload list of top 100 frequent drugs generated from the training data
    drug_by_category_output_txt_file_location: the output file location and name, eg: /home/huangz36/med_table_pivot_by_drug_category_and_counts.txt
    '''
    df_med_dedup = df_med.drop_duplicates()
    df_med_earliest_latest = df_med_dedup.groupby(['patientid'])[['days_to_covid_diag']].agg(['min','max']).reset_index().drop_duplicates()
    df_med_earliest_latest.columns = ['patientid', 'earliest_med_response_to_covid_diag', 'latest_med_response_to_covid_diag'] 
    # df_med_earliest_latest.to_csv(days_to_response_output_txt_file_location, sep='\t', index=False, header=True)  

    ## Make the table with drugs groups by categories
    ## Read in df with top 100 drugs, columns: ()
    top100_drug_list = df_top_100_drug['Drug'].to_list()
    df_med_dedup_top100_drugs = df_med_dedup[df_med_dedup['drug_name'].isin(top100_drug_list)]
    df_med_dedup_top100_drugs_and_cat = df_med_dedup_top100_drugs.merge(df_top_100_drug, how='left', left_on='drug_name', right_on='Drug')
    df_med_top100_drug_by_category_agg = df_med_dedup_top100_drugs_and_cat.groupby(['patientid', 'Category'])[['drug_name']].count().reset_index()
    df_med_top100_drug_by_category_agg.columns = ['patientid', 'Category', 'Drug_count']
    df_med_top100_drug_pivot_table = df_med_top100_drug_by_category_agg.pivot(index=["patientid"], \
                         columns=["Category"], values=['Drug_count']).fillna(0.0).reset_index() 
    ### Reassign column names
    new_column_names = []
    for (row1, row2) in df_med_top100_drug_pivot_table.columns:
        if row1 == 'patientid':
            column_name = str(row1)
        else:
            column_name = str(row1) + "_" + str(row2)
        new_column_names.append(column_name)
    df_med_top100_drug_pivot_table.columns = new_column_names
    ### Deal with "count" values, turn "count" into 1.0
    df_ID_col = df_med_top100_drug_pivot_table["patientid"].to_frame()   
    df_med_top100_drug_pivot_table_rest =  df_med_top100_drug_pivot_table.drop(columns=["patientid"])
    df_med_top100_drug_pivot_table_rest = df_med_top100_drug_pivot_table_rest.where(df_med_top100_drug_pivot_table_rest==0.0, other=1.0)
    df_med_top100_drug_pivot_table_reorg = pd.concat([df_ID_col,df_med_top100_drug_pivot_table_rest], axis=1)  
    # df_med_top100_drug_pivot_table_reorg.to_csv(drug_by_category_output_txt_file_location, sep='\t', index=False, header=True)    
    return (df_med_earliest_latest, df_med_top100_drug_pivot_table)


def clean_up_lab_table_Sharon_method(df_lab, top_50_lab_test_file, output_txt_file_location):
    '''
    Use Sharon's method to clean up lab table, for top 50 measurements
    1 for out of normal range, 0 for within normal range.
    If multiple measurements for the same patient, take the mean.
    Output number between 0 and 1, the larger, the more likely to have abnormal values
    '''
    df_lab_dedup = df_lab.drop_duplicates()
    ## read in top 50 labtests
    top_50_lab_type_list = read_txt_to_list(top_50_lab_test_file)
    df_lab_top_tests = df_lab_dedup[df_lab_dedup['test_name'].isin(top_50_lab_type_list)]
    print("df_lab_top_50_tests shape", df_lab_top_tests.shape)
    
    ## string process
    for i, row in df_lab_top_tests.iterrows():
        # print("row", row)
        if (row.normal_range is None) or (row.normal_range == '-') or (row.test_result == 'unknown'):
            df_lab_top_tests.at[i,'value_normal'] = 'None'
        elif (row.normal_range == row.test_result) :
            df_lab_top_tests.at[i,'value_normal'] = 0
        else:
            range_value = row.normal_range.replace("-",",")
            value = re.findall(r"[-+]?(?:\d*\.*\d+)", range_value)
            try:
                if (len(value) == 2) and (float(row.test_result) >= float(value[0])) and (float(row.test_result) <= float(value[1])): #the results in the normal range
                    df_lab_top_tests.at[i,'value_normal'] = 0
                else:
                    df_lab_top_tests.at[i,'value_normal'] = 1
                if (len(value) == 1):
                    x = re.findall("[<=>]", range_value)
                    s = "".join(x)
                    if (s == ">=") and (float(row.test_result) >= float(value[0])):
                        df_lab_top_tests.at[i,'value_normal'] = 0
                    elif (s == "<=") and (float(row.test_result) <= float(value[0])):
                        df_lab_top_tests.at[i,'value_normal'] = 0
                    elif (s == "<") and (float(row.test_result) < float(value[0])):
                        df_lab_top_tests.at[i,'value_normal'] = 0
                    elif (s == ">") and (float(row.test_result) > float(value[0])):
                        df_lab_top_tests.at[i,'value_normal'] = 0
                    else:
                        df_lab_top_tests.at[i,'value_normal'] = 1
            except (ValueError, TypeError):
                df_lab_top_tests.at[i,'value_normal'] = "Trash"
            
      ## generated new_table, and further process
    value_inlist = [0, 1]
    new_table_inlist = df_lab_top_tests[df_lab_top_tests['value_normal'].isin(value_inlist)]
    new_table_inlist_pivot = new_table_inlist.pivot_table(index=["patientid"], columns=["test_name"], values=['value_normal'], aggfunc={'value_normal': np.mean})
    value_other = ['None']
    new_table_other = df_lab_top_tests[df_lab_top_tests['value_normal'].isin(value_other)]
    value_trash = ['Trash']
    new_table_trash = df_lab_top_tests[df_lab_top_tests['value_normal'].isin(value_trash)]
    # new_table_inlist_pivot.to_csv(output_txt_file_location, sep='\t', index=True, header=True)
    
    return new_table_inlist_pivot

def clean_up_lab_table(df_lab, top_40_lab_test_file, output_txt_file_location, days_to_response_output_txt_file_location):
    '''
    Read in df_lab table
    Preload Top40 labtests
    In the function, pivot each lab test by taking max or min of multiple results
    '''
    df_lab_dedup = df_lab.drop_duplicates()
    df_lab_dedup_noNA = df_lab_dedup[df_lab_dedup['test_result']!='unknown']
    ## read in top 40 labtests
    top_40_lab_type_list = read_txt_to_list(top_40_lab_test_file)
    df_lab_top_tests = df_lab_dedup_noNA[df_lab_dedup_noNA['test_name'].isin(top_40_lab_type_list)]
    ## Pivot the top 40 lab tests one by one
    ### OxSatu min
    df_lab_top_tests_OxSatu = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Oxygen saturation (SpO2).pulse oximetry'].dropna()
    df_lab_top_tests_OxSatu_value = df_lab_top_tests_OxSatu[df_lab_top_tests_OxSatu['test_result'].str.contains('\d')]
    df_lab_top_tests_OxSatu_nonerror_value = df_lab_top_tests_OxSatu_value[df_lab_top_tests_OxSatu_value['test_result'].astype(float)<=100]
    df_lab_top_tests_OxSatu_nonerror_value['test_result'] = df_lab_top_tests_OxSatu_nonerror_value['test_result'].astype(float)
    df_lab_top_tests_OxSatu_min_mean = df_lab_top_tests_OxSatu_nonerror_value.groupby('patientid')[['test_result']].agg(['min','mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_OxSatu_min_mean.columns = ['patientid', 'OxSatu_min', 'OxSatu_mean']

    ### Glucose max
    df_lab_top_tests_Glucose = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Glucose.random'].dropna()
    df_lab_top_tests_Glucose_value = df_lab_top_tests_Glucose[df_lab_top_tests_Glucose['test_result'].str.contains('\d')]
    df_lab_top_tests_Glucose_value['test_result'] = df_lab_top_tests_Glucose_value['test_result'].astype(float)
    df_lab_top_tests_Glucose_max_mean = df_lab_top_tests_Glucose_value.groupby('patientid')[['test_result']].\
    agg(['max','mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Glucose_max_mean.columns = ['patientid', 'Glucose_max', 'Glucose_mean']

    ### Creatinine max
    df_lab_top_tests_Creatinine = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Creatinine'].dropna()
    df_lab_top_tests_Creatinine_value = df_lab_top_tests_Creatinine[pd.to_numeric(df_lab_top_tests_Creatinine.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Creatinine_value['test_result'] = df_lab_top_tests_Creatinine_value['test_result'].astype(float)
    df_lab_top_tests_Creatinine_max_mean = df_lab_top_tests_Creatinine_value.groupby('patientid')[['test_result']].\
    agg(['max','mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Creatinine_max_mean.columns = ['patientid', 'Creatinine_max', 'Creatinine_mean']

    ### WBC max
    df_lab_top_tests_WBC = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='White blood cell count (WBC)'].dropna()
    df_lab_top_tests_WBC_value = df_lab_top_tests_WBC[pd.to_numeric(df_lab_top_tests_WBC.test_result, errors='coerce').notna()]
    df_lab_top_tests_WBC_value['test_result'] = df_lab_top_tests_WBC_value['test_result'].astype(float)
    df_lab_top_tests_WBC_max_mean = df_lab_top_tests_WBC_value.groupby('patientid')[['test_result']]\
    .agg(['max','mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_WBC_max_mean.columns = ['patientid', 'WBC_max', 'WBC_mean']

    ### RBC max
    df_lab_top_tests_RBC = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Red blood cell count (RBC)'].dropna()
    df_lab_top_tests_RBC_value = df_lab_top_tests_RBC[pd.to_numeric(df_lab_top_tests_RBC.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_RBC_value['test_result'] = df_lab_top_tests_RBC_value['test_result'].astype(float)
    df_lab_top_tests_RBC_max_mean = df_lab_top_tests_RBC_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_RBC_max_mean.columns = ['patientid', 'RBC_max', 'RBC_mean']

    ### PLT max
    df_lab_top_tests_PLT = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Platelet count (PLT)'].dropna()
    df_lab_top_tests_PLT_value = df_lab_top_tests_PLT[pd.to_numeric(df_lab_top_tests_PLT.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_PLT_value['test_result'] = df_lab_top_tests_PLT_value['test_result'].astype(float)
    df_lab_top_tests_PLT_max_mean = df_lab_top_tests_PLT_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_PLT_max_mean.columns = ['patientid', 'PLT_max', 'PLT_mean']

    ### CO2 max
    df_lab_top_tests_CO2 = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Carbon dioxide.total (CO2)'].dropna()
    df_lab_top_tests_CO2_value = df_lab_top_tests_CO2[pd.to_numeric(df_lab_top_tests_CO2.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_CO2_value['test_result'] = df_lab_top_tests_CO2_value['test_result'].astype(float)
    df_lab_top_tests_CO2_max_mean = df_lab_top_tests_CO2_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_CO2_max_mean.columns = ['patientid', 'CO2_max', 'CO2_mean']

    ### Neutro perc
    df_lab_top_tests_Neutro = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Neutrophil.percent'].dropna()
    df_lab_top_tests_Neutro_value = df_lab_top_tests_Neutro[pd.to_numeric(df_lab_top_tests_Neutro.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Neutro_value['test_result'] = df_lab_top_tests_Neutro_value['test_result'].astype(float)
    df_lab_top_tests_Neutro_max_mean = df_lab_top_tests_Neutro_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Neutro_max_mean.columns = ['patientid', 'Neutro_max', 'Neutro_mean']

    ### lympho perc
    df_lab_top_tests_Lympho = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Lymphocyte.percent'].dropna()
    df_lab_top_tests_Lympho_value = df_lab_top_tests_Lympho[pd.to_numeric(df_lab_top_tests_Lympho.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Lympho_value['test_result'] = df_lab_top_tests_Lympho_value['test_result'].astype(float)
    df_lab_top_tests_Lympho_max_mean = df_lab_top_tests_Lympho_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Lympho_max_mean.columns = ['patientid', 'Lympho_max', 'Lympho_mean']

    ### Mono perc
    df_lab_top_tests_Mono = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Monocyte.percent'].dropna()
    df_lab_top_tests_Mono_value = df_lab_top_tests_Mono[pd.to_numeric(df_lab_top_tests_Mono.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Mono_value['test_result'] = df_lab_top_tests_Mono_value['test_result'].astype(float)
    df_lab_top_tests_Mono_max_mean = df_lab_top_tests_Mono_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Mono_max_mean.columns = ['patientid', 'Mono_max', 'Mono_mean']

    ### Eosin perc
    df_lab_top_tests_Eosin = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Eosinophil.percent'].dropna()
    df_lab_top_tests_Eosin_value = df_lab_top_tests_Eosin[pd.to_numeric(df_lab_top_tests_Eosin.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Eosin_value['test_result'] = df_lab_top_tests_Eosin_value['test_result'].astype(float)
    df_lab_top_tests_Eosin_max_mean = df_lab_top_tests_Eosin_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Eosin_max_mean.columns = ['patientid', 'Eosin_max', 'Eosin_mean']


    ### Baso perc
    df_lab_top_tests_Baso = df_lab_top_tests[df_lab_top_tests\
                                                       ['test_name']=='Basophil.percent'].dropna()
    df_lab_top_tests_Baso_value = df_lab_top_tests_Baso[pd.to_numeric(df_lab_top_tests_Baso.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Baso_value['test_result'] = df_lab_top_tests_Baso_value['test_result'].astype(float)
    df_lab_top_tests_Baso_max_mean = df_lab_top_tests_Baso_value.groupby('patientid')[['test_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Baso_max_mean.columns = ['patientid', 'Baso_max', 'Baso_mean']

    ## Add Chloride
    df_lab_top_tests_Chloride = df_lab_top_tests[df_lab_top_tests
                                                           ['test_name']=='Chloride (Cl)'].dropna()
    df_lab_top_tests_Chloride_value = df_lab_top_tests_Chloride[pd.to_numeric(df_lab_top_tests_Chloride.test_result, \
                                                                                      errors='coerce').notna()]
    df_lab_top_tests_Chloride_value['test_result'] = df_lab_top_tests_Chloride_value['test_result'].astype(float)
    df_lab_top_tests_Chloride_max_mean = df_lab_top_tests_Chloride_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Chloride_max_mean.columns = ['patientid', 'Chloride_max', 'Chloride_mean']

    ## Add MCH
    df_lab_top_tests_MCH = df_lab_top_tests[df_lab_top_tests
                                                           ['test_name']=='Mean corpuscular hemoglobin (MCH)'].dropna()
    df_lab_top_tests_MCH_value = df_lab_top_tests_MCH[pd.to_numeric(df_lab_top_tests_MCH.test_result, \
                                                                                      errors='coerce').notna()]
    df_lab_top_tests_MCH_value['test_result'] = df_lab_top_tests_MCH_value['test_result'].astype(float)

    df_lab_top_tests_MCH_max_mean = df_lab_top_tests_MCH_value.groupby('patientid')[['test_result']]\
    .agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_MCH_max_mean.columns = ['patientid', 'MCH_max', 'MCH_mean']

    ## Add Billirubin
    df_lab_top_tests_Billi = df_lab_top_tests[df_lab_top_tests
                                                           ['test_name']=='Bilirubin.total'].dropna()
    df_lab_top_tests_Billi_value = df_lab_top_tests_Billi[pd.to_numeric(df_lab_top_tests_Billi.test_result, \
                                                                                  errors='coerce').notna()]
    df_lab_top_tests_Billi_value['test_result'] = df_lab_top_tests_Billi_value['test_result'].astype(float)
    df_lab_top_tests_Billi_max_mean = df_lab_top_tests_Billi_value.groupby('patientid')[['test_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_lab_top_tests_Billi_max_mean.columns = ['patientid', 'Billi_max', 'Billi_mean']





    ### Aggregate all tables
    Labs_table_list = [df_lab_top_tests_OxSatu_min_mean, \
                   df_lab_top_tests_Glucose_max_mean, \
                   df_lab_top_tests_Creatinine_max_mean, \
                   df_lab_top_tests_WBC_max_mean, \
                   df_lab_top_tests_RBC_max_mean, \
                   df_lab_top_tests_PLT_max_mean, \
                   df_lab_top_tests_CO2_max_mean, \
                   df_lab_top_tests_Neutro_max_mean, \
                   df_lab_top_tests_Lympho_max_mean, \
                   df_lab_top_tests_Mono_max_mean, \
                   df_lab_top_tests_Eosin_max_mean, \
                   df_lab_top_tests_Baso_max_mean, \
                   df_lab_top_tests_Chloride_max_mean, \
                   df_lab_top_tests_MCH_max_mean, \
                   df_lab_top_tests_Billi_max_mean]
    df_lab_pid_for_left_merge = df_lab_top_tests[['patientid']].drop_duplicates()
    df_lab_pid_for_left_merge['status'] ='PT_with_LAB'
    ## left merge
    df = df_lab_pid_for_left_merge
    for obs_ind_table in Labs_table_list:
        df = df.merge(obs_ind_table, on='patientid', how='left').sort_values(by='patientid')
    df_lab_pivot = df.drop(['status'], axis=1)
    # df_lab_pivot.to_csv(output_txt_file_location, sep='\t', index=False, header=True)

    df_lab_earliest_latest = df_lab_dedup_noNA.groupby(['patientid'])[['days_to_covid_diag']].agg(['min','max']).reset_index().drop_duplicates()
    df_lab_earliest_latest.columns = ['patientid', 'earliest_lab_response_to_covid_diag', 'latest_lab_response_to_covid_diag']
    # df_lab_earliest_latest.to_csv(days_to_response_output_txt_file_location, sep='\t', index=False, header=True)

    return (df_lab_earliest_latest, df_lab_pivot)


def clean_up_obs_table(df_obs, output_txt_file_location, days_to_response_output_txt_file_location, top15_observation_list_location):
    '''
    df_obs: the name of the raw pandas dataframe read from the read_parquet
    '''
    df_obs_dedup = df_obs.drop_duplicates()
    ## Read in the top 15 obs labels
    top_15_obs_type_list = read_txt_to_list(top15_observation_list_location)
    df_obs_dedup_topObs = df_obs_dedup[df_obs_dedup['obs_type'].isin(top_15_obs_type_list)]
    df_obs_dedup_topObs['obs_type_unit'] = df_obs_dedup_topObs['obs_type'].astype(str) + "(" + df_obs_dedup_topObs['obs_unit'].astype(str) + ")"
    df_obs_dedup_topObs_combined = df_obs_dedup_topObs[['patientid', 'obs_type_unit', 'obs_result', 'days_to_covid_diag']]
    ## Pivot table by specific obs

    ## BMI max
    df_obs_dedup_topObs_BMI = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='BMI(None)'].dropna()
    df_obs_dedup_topObs_BMI_value = df_obs_dedup_topObs_BMI[pd.to_numeric(df_obs_dedup_topObs_BMI.obs_result, \
                                                                                  errors='coerce').notna()]
    df_obs_dedup_topObs_BMI_value['obs_result'] = df_obs_dedup_topObs_BMI_value['obs_result'].astype(float)
    df_obs_patient_BMI_max = df_obs_dedup_topObs_BMI_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_patient_BMI_max.columns = ['patientid', 'BMI_max', 'BMI_mean']


    ## HT max
    df_obs_dedup_topObs_HT = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='HT(cm)'].dropna()
    df_obs_dedup_topObs_HT_value = df_obs_dedup_topObs_HT[pd.to_numeric(df_obs_dedup_topObs_HT.obs_result,
                                                                        errors='coerce').notna()]
    df_obs_dedup_topObs_HT_value['obs_result'] = df_obs_dedup_topObs_HT_value['obs_result'].astype(float)    
    df_obs_dedup_topObs_HT_max = df_obs_dedup_topObs_HT_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_HT_max.columns = ['patientid', 'HT_max', 'HT_mean']

    ## WT max
    df_obs_dedup_topObs_WT = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='WT(kg)'].dropna()
    df_obs_dedup_topObs_WT_value = df_obs_dedup_topObs_WT[pd.to_numeric(df_obs_dedup_topObs_WT.obs_result,
                                                                        errors='coerce').notna()]                          
    df_obs_dedup_topObs_WT_value['obs_result'] = df_obs_dedup_topObs_WT_value['obs_result'].astype(float)
    df_obs_dedup_topObs_WT_max = df_obs_dedup_topObs_WT_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_WT_max.columns = ['patientid', 'WT_max', 'WT_mean']

    ## Pulse max
    df_obs_dedup_topObs_PULSE = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='PULSE(bpm)'].dropna()
    df_obs_dedup_topObs_PULSE_value = df_obs_dedup_topObs_PULSE[pd.to_numeric(df_obs_dedup_topObs_PULSE.obs_result,
                                                                        errors='coerce').notna()] 
    df_obs_dedup_topObs_PULSE_value['obs_result'] = df_obs_dedup_topObs_PULSE_value['obs_result'].astype(float)
    df_obs_dedup_topObs_PULSE_max = df_obs_dedup_topObs_PULSE_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_PULSE_max.columns = ['patientid', 'PULSE_max', 'PULSE_mean']

    ## Temp max
    df_obs_dedup_topObs_TEMP = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='TEMP(deg c)'].dropna()
    df_obs_dedup_topObs_TEMP_value = df_obs_dedup_topObs_TEMP[pd.to_numeric(df_obs_dedup_topObs_TEMP.obs_result,
                                                                        errors='coerce').notna()]                               
    df_obs_dedup_topObs_TEMP_value['obs_result'] = df_obs_dedup_topObs_TEMP_value['obs_result'].astype(float)
    df_obs_dedup_topObs_TEMP_max = df_obs_dedup_topObs_TEMP_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_TEMP_max.columns = ['patientid', 'TEMP_max', 'TEMP_mean']

    ## Resp max
    df_obs_dedup_topObs_RESP = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='RESP(breaths/min)'].dropna()
    df_obs_dedup_topObs_RESP_value = df_obs_dedup_topObs_RESP[pd.to_numeric(df_obs_dedup_topObs_RESP.obs_result,
                                                                        errors='coerce').notna()]
    df_obs_dedup_topObs_RESP_value['obs_result'] = df_obs_dedup_topObs_RESP_value['obs_result'].astype(float)
    df_obs_dedup_topObs_RESP_max = df_obs_dedup_topObs_RESP_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_RESP_max.columns = ['patientid', 'RESP_max', 'RESP_mean']
    ## HR max
    df_obs_dedup_topObs_HR = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='HR(bpm)'].dropna()
    df_obs_dedup_topObs_HR_value = df_obs_dedup_topObs_HR[pd.to_numeric(df_obs_dedup_topObs_HR.obs_result,
                                                                        errors='coerce').notna()]            
    df_obs_dedup_topObs_HR_value['obs_result'] = df_obs_dedup_topObs_HR_value['obs_result'].astype(float)
    df_obs_dedup_topObs_HR_max = df_obs_dedup_topObs_HR_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_HR_max.columns = ['patientid', 'HR_max', 'HR_mean']

    ## Deal with categorical results, drop inconsistent records
    ### Smoke
    df_obs_dedup_topObs_SMOKE = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined['obs_type_unit']=='SMOKE(None)']\
    [['patientid', 'obs_result']].drop_duplicates().reset_index(drop=True)
    df_obs_dedup_topObs_SMOKE.columns = ['patientid', 'SMOKE']
    df_obs_smoke_count = df_obs_dedup_topObs_SMOKE.groupby('patientid')[['SMOKE']].count().\
    reset_index().sort_values(by='SMOKE', ascending=False)
    smoke_inconsistent_to_drop = df_obs_smoke_count[df_obs_smoke_count['SMOKE']!=1]['patientid'].to_list()
    df_obs_dedup_topObs_SMOKE_dedup = df_obs_dedup_topObs_SMOKE[~df_obs_dedup_topObs_SMOKE['patientid'].\
                                                      isin(smoke_inconsistent_to_drop)]
    ### Alcohol                                                  
    df_obs_dedup_topObs_ALCOHOL = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='ALCOHOL(None)'].dropna()[['patientid', 'obs_result']].drop_duplicates().reset_index(drop=True)
    df_obs_dedup_topObs_ALCOHOL.columns = ['patientid', 'ALCOHOL']
    df_obs_alcohol_count = df_obs_dedup_topObs_ALCOHOL.groupby('patientid')[['ALCOHOL']].count().\
    reset_index().sort_values(by='ALCOHOL', ascending=False)
    alcohol_inconsistent_to_drop = df_obs_alcohol_count[df_obs_alcohol_count['ALCOHOL']!=1]['patientid'].to_list()
    df_obs_dedup_topObs_ALCOHOL_dedup = df_obs_dedup_topObs_ALCOHOL[~df_obs_dedup_topObs_ALCOHOL['patientid'].\
                                                      isin(alcohol_inconsistent_to_drop)]
    ### Pain
    df_obs_dedup_topObs_PAIN = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='PAIN(out of 10)'].dropna()
    df_obs_dedup_topObs_PAIN_value = df_obs_dedup_topObs_PAIN[pd.to_numeric(df_obs_dedup_topObs_PAIN.obs_result,
                                                                        errors='coerce').notna()] 
    df_obs_dedup_topObs_PAIN_value['obs_result'] = df_obs_dedup_topObs_PAIN_value['obs_result'].astype(float)
    df_obs_dedup_topObs_PAIN_max = df_obs_dedup_topObs_PAIN_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_PAIN_max.columns = ['patientid', 'PAIN_max', 'PAIN_mean']
    ### PACKYRS
    df_obs_dedup_topObs_PACKYRS = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='PACK_YEARS(pack-years)'].dropna()
    df_obs_dedup_topObs_PACKYRS_value = df_obs_dedup_topObs_PACKYRS[pd.to_numeric(df_obs_dedup_topObs_PACKYRS.obs_result,
                                                                        errors='coerce').notna()] 
    df_obs_dedup_topObs_PACKYRS_value['obs_result'] = df_obs_dedup_topObs_PACKYRS_value['obs_result'].astype(float)
    df_obs_dedup_topObs_PACKYRS_max = df_obs_dedup_topObs_PACKYRS_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_PACKYRS_max.columns = ['patientid', 'PACKYRS_max', 'PACKYRS_mean']
    ### DBP
    df_obs_dedup_topObs_DBP = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='DBP(mm Hg)'].dropna()
    df_obs_dedup_topObs_DBP_value = df_obs_dedup_topObs_DBP[pd.to_numeric(df_obs_dedup_topObs_DBP.obs_result,
                                                                        errors='coerce').notna()] 
    df_obs_dedup_topObs_DBP_value['obs_result'] = df_obs_dedup_topObs_DBP_value['obs_result'].astype(float)
    df_obs_dedup_topObs_DBP_max = df_obs_dedup_topObs_DBP_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_DBP_max.columns = ['patientid', 'DBP_max', 'DBP_mean']
    ### SBP
    df_obs_dedup_topObs_SBP = df_obs_dedup_topObs_combined[df_obs_dedup_topObs_combined\
                                                       ['obs_type_unit']=='SBP(mm Hg)'].dropna()
    df_obs_dedup_topObs_SBP_value = df_obs_dedup_topObs_SBP[pd.to_numeric(df_obs_dedup_topObs_SBP.obs_result,
                                                                        errors='coerce').notna()] 
    df_obs_dedup_topObs_SBP_value['obs_result'] = df_obs_dedup_topObs_SBP_value['obs_result'].astype(float)
    df_obs_dedup_topObs_SBP_max = df_obs_dedup_topObs_SBP_value.groupby('patientid')[['obs_result']].agg(['max', 'mean']).reset_index().sort_values(by='patientid')
    df_obs_dedup_topObs_SBP_max.columns = ['patientid', 'SBP_max', 'SBP_mean']
    ## Aggregate all tables
    obs_table_list = [df_obs_patient_BMI_max, df_obs_dedup_topObs_HT_max, df_obs_dedup_topObs_WT_max, \
                 df_obs_dedup_topObs_PULSE_max, df_obs_dedup_topObs_TEMP_max, df_obs_dedup_topObs_RESP_max, \
                 df_obs_dedup_topObs_HR_max, \
                 df_obs_dedup_topObs_PAIN_max, df_obs_dedup_topObs_PACKYRS_max, \
                 df_obs_dedup_topObs_DBP_max, df_obs_dedup_topObs_SBP_max, \
                 df_obs_dedup_topObs_SMOKE_dedup, df_obs_dedup_topObs_ALCOHOL_dedup]
    df_obs_pid_for_left_merge = df_obs_dedup_topObs_combined[['patientid']].drop_duplicates()
    df_obs_pid_for_left_merge['status'] ='PT_with_OBS'
    df = df_obs_pid_for_left_merge
    for obs_ind_table in obs_table_list:
        df = df.merge(obs_ind_table, on='patientid', how='left').sort_values(by='patientid')
    df_obs_pivot_by_top15_obs = df.drop(['status'], axis=1)
    # df_obs_pivot_by_top15_obs.to_csv(output_txt_file_location, sep='\t', index=False, header=True)


    ## Make the table with earliest and latest "days to response"
    '''
    df_obs_earliest_response = df_obs_dedup.groupby(['patientid'])[['days_to_covid_diag']].min().reset_index()
    df_obs_earliest_response.columns = [['patientid', 'earliest_obs_response_to_covid_diag']]
    df_obs_latest_response = df_obs_dedup.groupby(['patientid'])[['days_to_covid_diag']].min().reset_index()
    df_obs_latest_response.columns = [['patientid', 'latest_obs_response_to_covid_diag']]
    df_obs_earliest_latest = df_obs_earliest_response.merge(df_obs_latest_response, on='patientid', how='inner').sort_values(by='patientid')
    df_obs_earliest_latest.to_csv(days_to_response_output_txt_file_location, sep='\t', \
                                index=False, header=True)
    '''
    df_obs_earliest_latest = df_obs_dedup.groupby(['patientid'])[['days_to_covid_diag']].agg(['min','max']).reset_index().drop_duplicates()
    df_obs_earliest_latest.columns = ['patientid', 'earliest_obs_response_to_covid_diag', 'latest_obs_response_to_covid_diag']
    # df_obs_earliest_latest.to_csv(days_to_response_output_txt_file_location, sep='\t', index=False, header=True)   
    return (df_obs_earliest_latest, df_obs_pivot_by_top15_obs)
    
    
    
def process_dem_table(df_dem, output_txt_file_location):
    ## Create datetime64 object for YYYY-MM-DD
    timezone = tz.tzlocal()
    date_covid_breakout = pd.to_datetime(np.datetime64('2020-01-01')).replace(tzinfo=timezone)
    date_long_covid = pd.to_datetime(np.datetime64('2021-06-01')).replace(tzinfo=timezone)
    
    # df_dem['index_month_year'] = pd.to_datetime(df_dem['index_month_year'])
    
    ## Process dates
    df_dem['index_month_year_from_begin'] = df_dem['index_month_year'] - date_covid_breakout
    df_dem['after_long_covid_start'] = df_dem['index_month_year'] > date_long_covid
    ## Fill in the middle of age when birth_yr missing
    df_dem['birth_yr_processed'] = df_dem['birth_yr']
    df_dem['birth_yr_processed'] = df_dem['birth_yr_processed'].fillna(1968)
    df_dem.loc[df_dem['birth_yr_processed'].str.contains('Earlier', na=False), 'birth_yr_processed'] = 1932
    df_dem.loc[df_dem['birth_yr_processed'].str.contains('Un', na=False), 'birth_yr_processed'] = 1968 # median age
    df_dem['age'] = 2021 - df_dem['birth_yr_processed'].astype(int)
    df_dem['index_month_year_from_begin'] =  df_dem['index_month_year_from_begin'].dt.days
    df_dem['after_long_covid_start'] =  df_dem['after_long_covid_start'].replace({True: 1, False: 0})    
    df_dem_processed = df_dem[['patientid', 'birth_yr', 'gender', 'race', 'ethnicity',\
                                  'index_month_year', 'index_month_year_from_begin', \
                                  'after_long_covid_start','age']]   
    # df_dem_processed.to_csv(output_txt_file_location, sep='\t', index=False, header=True)

    return (df_dem_processed)


def process_imm_table(df_imm, output_txt_file_location, days_to_response_output_txt_file_location):
    '''
    Process imm table by selecting meaningful vaccines
    '''
    df_imm_dedup = df_imm.drop_duplicates()
    df_imm_dedup['immunization_desc'] = df_imm_dedup['immunization_desc'].str.replace(';', ',').str.upper()
    ## Drop na for 'immunization_desc' and 'mapped_name' columns
    df_imm_cleaned = df_imm_dedup.dropna(subset=['mapped_name', 'immunization_desc'])
    df_imm_cleaned = df_imm_cleaned[df_imm_cleaned['immunization_desc']!='UNSPECIFIED IMMUNIZATION']
    
    ## String process for vaccine names
    cnt = 0 # record the number of NoneType mapped_name is processed
    desc_list = [] # record the immunization_desc that is not categorized and directly copied into mapped_name 
    for i, row in df_imm_cleaned.iterrows():
        if row.mapped_name is None:
            cnt += 1
            desc = row.immunization_desc

            # fill in the mapped_name information for COVID vaccine
            if 'COVID-19' in desc or 'SARS-COV-2' in desc or 'COVID' in desc or 'COV-19' in desc:

                # add MODERNA or PFIZER based on different doses
                if '100MCG/0.5ML' in desc.replace(' ','') or 'MRNA-1273' in desc or 'MRNA 1273' in desc or '100 MCG OR 50 MCG DOSE' in desc:
                #eg 'SARS-COV-2 (COVID-19) VACCINE, MRNA, SPIKE PROTEIN, LNP, PRESERVATIVE FREE, 100 MCG/0.5ML DOSE':
                    desc = desc + ' (MODERNA)'
                    df_imm_cleaned.at[i,'immunization_desc'] = desc
                if '30MCG/0.3ML' in desc.replace(' ',''):
                #eg. 'SARS-COV-2 (COVID-19) VACCINE, MRNA, SPIKE PROTEIN, LNP, PRESERVATIVE FREE, 30 MCG/0.3ML DOSE':
                    desc = desc + '(PFIZER)'
                    df_imm_cleaned.at[i,'immunization_desc'] = desc

                # fill in mapped_name column based on immunization_desc
                if 'JANSSEN' in desc or 'JNJ' in desc or 'JSN' in desc or 'JOHNSON AND JOHNSON' in desc or 'AD26' in desc:
                    df_imm_cleaned.at[i,'mapped_name'] = 'COVID-19 vac, Ad26.COV2.S (Janssen)/PF'

                elif 'PFIZER' in desc or 'MODERNA' in desc or 'MRNA BNT-162B2' in desc or 'NOVAVAX' in desc:
                    df_imm_cleaned.at[i,'mapped_name'] = 'COVID-19 vaccine, mRNA, BNT162b2, LNP-S (Pfizer)/PF'

                elif 'RS-CHADOX1' in desc:
                    df_imm_cleaned.at[i,'mapped_name'] = 'COVID-19 vaccine, AZD-1222 (AstraZeneca)/PF'

                elif 'NON-US' in desc or 'SINOVAC' in desc:
                    df_imm_cleaned.at[i,'mapped_name'] = 'COVID-19, Non-US'

                else:
                    df_imm_cleaned.at[i,'mapped_name'] = 'COVID-19, Unspecified'

            # fill in the mapped_name information for Flu Shot
            elif 'INFLUENZA' in desc or 'FLU ' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'Flu Vaccine Unspecified'

            # other popular vaccines
            elif 'HEPATITIS B' in desc or 'HEP B' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'Hepatitis B Vaccine Unspecified'
            elif 'PNEUMOCOCCAL' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'Pneumococcal Vaccine - Unspecified'
            elif 'RABIES' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'Rabies Vaccine, Unspecified'
            elif 'ZOSTER' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'Zoster Vaccine, Unspecified'
            elif 'YELLOW FEVER' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'yellow fever vaccine live/PF'
            elif 'TYPHOID' in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'typhoid vacc,live,attenuated'
            elif 'RHO(D)' in desc.replace(' ','') or 'RHO D 'in desc:
                df_imm_cleaned.at[i,'mapped_name'] = 'RHO (D) IMMUNE GLOBULIN'
            elif 'TDAP' in desc.upper():
                df_imm_cleaned.at[i,'mapped_name'] = 'Acellular Pertussis-Diphtheria-Tetanus Toxoids Vaccine-Reduced (TDaP)'
            else:
                desc_list.append(row.immunization_desc)    
                df_imm_cleaned.at[i,'mapped_name'] = desc
    
    ## Make pivoted immu_new table
    df_immu_new = []
    patientIDs = df_imm_cleaned.patientid.unique().tolist()
    for pid in patientIDs:
        patient_data = df_imm_cleaned[df_imm_cleaned.patientid==pid]
        desc_list = patient_data.immunization_desc.tolist()
        name_list = patient_data.mapped_name.tolist()

        #entry = {'patientid':pid, 'days_to_covid_diag_min':days_min,'days_to_covid_diag_max':days_max,'immunization_desc': desc_list, 'mapped_name':name_list,'COVID-19':'','Influenza':'',\
                #'Pertussis':0, 'Diphtheria':0, 'Haemophilus B':0, 'Pneumococcal':0, 'MMR':0, 'HPV':0, 'VZV':0, 'Rabies':0, 'HEP A':0, 'HEP B':0, 'Palio':0}

        entry = {'patientid':pid, 'COVID-19':'','Influenza':'','Pertussis':0, 'Diphtheria':0, 'Haemophilus B':0, 'Pneumococcal':0, 'MMR':0, 'HPV':0, 'VZV':0, 'Rabies':0, 'HEP A':0, 'HEP B':0, 'Palio':0}

        covid = []
        for i in range(0,len(name_list)):
            name = name_list[i]
            #COVID column
            if 'COVID-19' in name:
                if name == 'COVID-19 vac, Ad26.COV2.S (Janssen)/PF':
                    covid_type = 'JNJ'
                elif name == 'COVID-19 vaccine, AZD-1222 (AstraZeneca)/PF':
                    covid_type = 'AstraZeneca'
                elif name == 'COVID-19, Non-US':
                    covid_type = 'Non-US'
                elif name == 'COVID-19, Unspecified':
                    covid_type = 'Unspecified'
                elif name == 'COVID-19 vaccine, mRNA, BNT162b2, LNP-S (Pfizer)/PF':
                    desc = desc_list[i]
                    if desc != None:
                        if 'MODERNA' in desc:
                            covid_type='MODERNA'
                        elif 'PFIZER' in desc:
                            covid_type='PFIZER'
                        elif 'NOVAVAX' in desc:
                            covid_type='NOVAVAX'
                        elif 'JANSSEN' in desc:
                            covid_type='JNJ'
                        else:
                            covid_type='Unspecified'
                else:
                    covid_type ='Unspecified'
                covid.append(covid_type)

            # FLU column
            # note: few cases where both "inactivated" and "live" flu vaccine are present for one patient, ignore and only report one in the new table
            if 'flu ' in name.lower() or 'influenza' in name.lower():
                if name in ['Influenza Live Vaccine', 'influenza vaccine quadrivalent live 2020-2021 (2 yrs-49 yrs)']:
                    entry['Influenza'] = 'Live'
                elif name in ['flu vaccine quad 2020-2021(4 years and older)cell derived/PF','Influenza Inactivated Vaccine']:
                    entry['Influenza'] = 'Inactivated'
                elif name == 'Flu Vaccine Unspecified':
                    entry['Influenza'] = 'Unspecified'
                else:
                    entry['Influenza'] = 'Unspecified'

            # vaccine against other infections: pertussis, diphtheria, Hib, MMRV, HEP A, HEP B, HPV, VZV, Rabies, Paliovirus
            if 'pertussis' in name.lower():
                entry['Pertussis'] = 1
            if 'diphtheria' in name.lower():
                entry['Diphtheria'] = 1
            if 'haemophilus' in name.lower():
                entry['Haemophilus B'] = 1   
            if 'pneumococcal' in name.lower():
                entry['Pneumococcal'] = 1   
            if 'measles' in name.lower() or 'mumps' in name.lower() or 'rubella' in name.lower():
                entry['MMR'] = 1  
            if 'papillomavirus' in name.lower():
                entry['HPV'] = 1
            if 'rabies' in name.lower():
                entry['Rabies'] = 1
            if 'zoster' in name.lower():
                entry['VZV'] = 1
            if 'hepatitis b' in name.lower():
                entry['HEP B'] = 1
            if 'hepatitis a' in name.lower():
                entry['HEP A'] = 1
            if 'palio' in name.lower():
                entry['Palio'] = 1  

        # print(covid)
        covid=list(set(covid)) # hard to know whether # occurrence equals how many doses a patient has taken, so keep this simple
        covid_str = ','.join(map(str,covid))
        entry['COVID-19'] = covid_str

        df_immu_new.append(entry)
        
    df_immu_new = pd.DataFrame(df_immu_new)
    
    print("pivoted imm table shape", df_immu_new.shape)
    # df_immu_new.to_csv(output_txt_file_location, sep='\t', index=False, header=True)
    
    # Earliest and latest days table
    df_imm_earliest_latest = df_imm_cleaned.groupby(['patientid'])[['days_to_covid_diag']].agg(['min','max']).reset_index().drop_duplicates()
    df_imm_earliest_latest.columns = ['patientid', 'earliest_imm_response_to_covid_diag', 'latest_imm_response_to_covid_diag']
    # print(df_imm_earliest_latest.head(5))
    # print(df_imm_earliest_latest.shape)
    # df_imm_earliest_latest.to_csv(days_to_response_output_txt_file_location, sep='\t', index=False, header=True)
    
    return (df_imm_earliest_latest, df_immu_new)
    
# def make_arg_parser():
    # ''' command line argument '''
    # parser = argparse.ArgumentParser(description='TMB caller at sample level.')
    # parser.add_argument("-o", "--outputdir", help="Output dir name", required = True)
    # parser.add_argument("-d", "--datadir", help="eg: /challenge/seeing-through-the-fog/data/train_data", required = True)
    # return parser



def processRawData(datadir, outputdir, experiment):
    # parser = make_arg_parser()
    # args = parser.parse_args()
    # outputdir = args.outputdir
    # datadir = args.datadir    
    # if not os.path.exists(outputdir):
    #     os.makedirs(outputdir)
    #     print("created outputdir")

    ## data locations
    #datadir = '/challenge/seeing-through-the-fog/data/train_data'
    df_dia = pd.read_parquet(datadir + "/" + "diagnoses.parquet")
    df_obs = pd.read_parquet(datadir + "/" + "observations.parquet")
    df_med = pd.read_parquet(datadir + "/" + "medication.parquet")
    df_lab = pd.read_parquet(datadir + "/" + "labs.parquet")
    df_imm = pd.read_parquet(datadir + "/" + "immunization.parquet")
    df_dem = pd.read_parquet(datadir + "/" + "demo.parquet")
    
    top_40_lab_test_file = "./preload/Top40_labtests.txt"
    top_50_lab_test_file = "./preload/Top50_labtests.txt"
    df_top_100_drug = pd.read_csv("./preload/Top100_drug_with_annotated_category.txt", sep='\t', header=0)
    top15_observation_list_location = "./preload/Top15_obs_type.txt"    

    ## clean up table by table
    df_dem_pivot = process_dem_table(df_dem, outputdir + "/" + "dem_with_age_yr_processed.txt")
    print("cleaned dem table")
    
    (df_dia_earliest_latest, df_dia_pivot) = clean_up_dia_table(df_dia, outputdir + "/" + "test_dia_table_pivot_by_ICD10_code.txt", outputdir + "/" + "dia_table_earliest_latest.txt")
    print("cleaned dia table")
    (df_med_earliest_latest, df_med_pivot) = clean_up_med_table(df_med, outputdir + "/" + "med_table_earliest_med_response_by_patientid.txt", df_top_100_drug, outputdir + "/" + "med_table_pivot_by_drug_category_and_counts.txt")
    print("cleaned med table")
    (df_lab_earliest_latest, df_lab_pivot) = clean_up_lab_table(df_lab, top_40_lab_test_file, outputdir + "/" + "lab_table_pivot.txt", outputdir + "/" + "lab_table_earliest_latest.txt")
    print("cleaned lab table")
    df_lab_pivot_sharon_method = clean_up_lab_table_Sharon_method(df_lab, top_50_lab_test_file, outputdir + "/" + "lab_table_pivot_Sharon_method.txt")
    print("cleaned lab table")
    (df_obs_earliest_latest, df_obs_pivot) = clean_up_obs_table(df_obs, outputdir + "/" + "obs_table_pivot_by_top_obs.txt", outputdir + "/" + "obs_table_earliest_latest.txt", top15_observation_list_location)
    print("cleaned obs table")
    
    (df_imm_earliest_latest, df_imm_pivot) = process_imm_table(df_imm, outputdir + "/" + "imm_table_pivot.txt", outputdir + "/" + "imm_table_earliest_latest.txt")

    # aggregate min/max days
    days_dfs = [df_dia_earliest_latest, df_med_earliest_latest, df_lab_earliest_latest, df_obs_earliest_latest, df_imm_earliest_latest]
    df_days_merged = reduce(lambda  left,right: pd.merge(left,right,on=['patientid'],how='outer'), days_dfs)
    print("columns of the merged table", df_days_merged.columns)
    df_days_merged['earliest_to_covid_diag'] = df_days_merged[["earliest_dia_response_to_covid_diag", "earliest_med_response_to_covid_diag", "earliest_lab_response_to_covid_diag", "earliest_obs_response_to_covid_diag", "earliest_imm_response_to_covid_diag"]].min(axis=1)
    df_days_merged['latest_to_covid_diag'] = df_days_merged[["latest_dia_response_to_covid_diag", "latest_med_response_to_covid_diag", "latest_lab_response_to_covid_diag", "latest_obs_response_to_covid_diag", "latest_imm_response_to_covid_diag"]].max(axis=1)
    df_days_merged = df_days_merged[['patientid', 'earliest_to_covid_diag', 'latest_to_covid_diag']]
    # df_days_merged.to_csv(outputdir + "/" + "earliest_latest_days.txt", index=False, header=True)
    # Both inner merge and outer merge to make master table
    tables_to_merge = [df_days_merged, df_dia_pivot, df_med_pivot, df_lab_pivot, df_lab_pivot_sharon_method, df_obs_pivot, df_dem_pivot, df_imm_pivot]
    mergedDF_inner = reduce(lambda left, right:
                     pd.merge(left, right,
                             on = ['patientid'], how = 'inner'),
                     tables_to_merge)
    print("mergedDF_inner shape",  mergedDF_inner.shape)   
    mergedDF_outer = reduce(lambda left, right:
                     pd.merge(left, right,
                             on = ['patientid'], how = 'outer'),
                     tables_to_merge)
    print("mergedDF_outer shape",  mergedDF_outer.shape)  
    # mergedDF_inner.to_csv(outputdir + "/" + experiment + "_master_table_inner_merge.txt", sep='\t', index=False, header=True)
    mergedDF_outer.to_csv(outputdir + "/" + experiment + "_master_table_outer_merge.txt", sep='\t', index=False, header=True)    
    
    if experiment.lower() == 'train':
        df_tar = pd.read_parquet(datadir + "/" + "target.parquet")
        df_tar.to_csv(outputdir + "/" + "target.txt", sep='\t', index=False, header=True)
        
        return mergedDF_inner, mergedDF_outer, df_tar
    
    return mergedDF_inner, mergedDF_outer
        
