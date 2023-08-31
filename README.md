# luminous

**Seeing through the fog**

* `Abid Hasan` (hasana1; US; Computational Science SCl; Diagnostics)

* `Julie Huang` (huangz36; US; Computational Science SCl; Diagnostics)

* `Ke Chen` (chenk62; US; Computational Science SCl; Diagnostics)

* `Sharon Chiu` (chiuh4; US; Computational Science SCl; Diagnostics)



Lead's email: abid.hasan@roche.com

## Team's summary for the expert panel

### Executive summary
**Preprocessing:** 
Preprocessing is performed on raw “parquet” files to extract features. 

For the `demo` table, along with other demographic information, we have observed that the detection time of COVID events is highly related to the probability of long COVID. Patients with `Index_month_year` after `2021-06-01` are more likely to develop long COVID (Figure 1), we have thus generated an integer feature to record the number of days of index_month_year from COVID first break out (2020-01-01).

![Long Covid Ratio](img/RAAD_Figure1.jpg?raw=true)

For the `labs` table, two major approaches were taken, one is to categorize each `test_result` as within normal range or out of normal range, the other is to aggregate the lab measurements taken over the time for the same patients (eg. mean of creatinine, mean of glucose and mean of red blood cell count).  

In the `observations` table, prevalent `obs_type` were extracted, and aggregated as “mean”, “minimum” or “maximum” for patients with multiple measurements over the time. 

The `diagnoses` and `medication` tables contain sparse information. We grouped diagnosis ICD codes by categories of encoded letters (https://www.cms.gov/medicare/icd-10/2023-icd-10-cm). Drug names that are within the top 100 prevalent in the `medication` table have been summarized into the following categories manually: antibiotics, anxiety, breath, cholesterol, cough, diabetes, fever_pain, heart_blood, hormon, immune, infection, mauscle, nause, nutrition, seizure, skin, stomach and others. 

The `immunization` table is the least well-documented data among all provided information. After cleaning up the table, we found only about 12% of the total patients (47897 out of all 395364 patients, and 192 out of the 1606 long-COVID patients) have provided immunization records. Among them, 22.1% (31.3% long-COVID) patients have received at least one type of COVID vaccine. We then carefully examined the consistency between the `immmunization_desc` and `mapped_name` columns, and after filling in the missing information, decided to use the `mapped_name` as the basis for further data extraction. Finally, we picked 10 frequently recorded vaccinations in addition to the COVID and Influenza vaccines to create the new immunization table. Drugs and vaccines recorded in less than 10 long-COVID patients were discarded for our purpose. The new immunization table includes the following vaccinations: COVID (categorized into “PFIZER”, ”MODERNA”, ”JNJ”, ”NOVAVAX”, “AstraZeneca”, ”Non-US”, and ”Unspecified”), Influenza (categorized into “Inactivated”, “Live”, and “Unspecified”), pertussis, diphtheria, haemophilus B, pneumococcal, MMR, HPV, rabies, VZV, hepatitis A, and hepatitis B.

**Feature engineering and processing:**
The following steps are taken in our feature engineering process. First, we examined each feature to determine its numerical, non-numerical (categorical), and missing contents. We found that when a feature has a combination of numerical and non-numerical values, the non-numeric data usually came from missing measurements and their number was insignificant. Therefore, for each column that had a mix of numeric and categorical data, we converted the categorical values into NaN values so that they are accepted by the feature selection and classification model. Second, the NaN values in all features are imputed with a median value of that column. We imputed with median value instead of mean to get a more representative value, because 1) most of the feature value distribution is highly skewed; and 2) for columns containing binary values, the mean of the column value may cause misrepresentation. Finally, for features with only categorical content, values were converted with one-hot encoding. One-hot-encoding is a process of creating dummy variables. This technique is used when the order of the categorical variables does not matter.

**Feature selection:**
For complex models, feature selection helps to improve prediction accuracy and model performance by selecting critical features and removing redundant ones. Before applying feature selection algorithms, it is important that we create a balanced dataset to avoid overfitting. The training data for this challenge is an extremely imbalanced dataset such that only 0.4% of all patients are diagnosed with long COVID. Hence, we randomly select 1606 short-COVID samples, so that we have the same size of both the majority and minority classes for feature selection.

Then, on the balanced dataset, we tried several different types of feature selection methods, i.e. the ridge, and lasso regression method, elastic net, and decision tree-based feature selection methods like a random forest. Considering the sparse nature of the training data, we did not choose any regression-based feature selection method, which can not handle zero values. For a similar reason, we avoided using the embedded feature selection methods, such as the elastic net, which uses an L1 and L2 penalty to shrink the coefficient. Finally, we used the random forest-based feature selection model, which tells us which feature in our data is most helpful toward the best classification. First, we optimized the hyperparameters of the random forest classifier that is used for the feature selection, and then the model is executed on the training data to obtain the features. The features are ranked based on their importance by the model. From a ranked list of features, we have selected the top 20 features for the prediction model. The selection of the top 20 features was determined by observing the best classification performances. Once the feature selection model selects the top features, we write the feature names into a file titled `top_features.txt`.

**Model selection and tuning:**
We explored several other classification models, including the Random Forest classifier, Gaussian Naive Bayes classifier, Logistic regression classifier, and Neural network-based classifier, i.e. Multilayer perceptron (MLP). For a complex and highly imbalanced dataset, a complex model for classification is warranted. For each of the classification models above, we performed hyperparameter tuning using a grid search cross-validation method. We chose XGBoost as our final prediction model. We have tuned several parameters for the XGBoost for optimal performance. The trained model is saved in a file named `trained_model.sav` that is used for the test data prediction.


### How we approached a rare outcome
This long covid dataset is highly imbalanced. In a vast majority of the negative samples, we are attributing positive predictions as rare outcomes. Based on our data processing and feature selection, we learn about a specific set of features that are significant for predicting rare outcomes. Although very few features ranked high during a feature selection, based on the prediction performance analysis we still believe all of the top set of features contribute to the prediction.

We devised a feature that calculates a time cutoff of when the long covid started, i.e. `after_long_covid_start`. It turned out to be the most important feature in our feature set. This selection of the feature makes sense since the patients before that time threshold are highly unlikely to develop long covid. Along with this feature, we considered patients' `age` as the next important feature. Although not exclusive, elder patients are more prone to long covid than younger patients. Along with these features, we considered some of their medical records and lab results which as a group helped the prediction to identify rare positive outcomes.

### Notes on novel predictors
Most of our predictors are processed data from the original data tables, i.e., “novel” in their information content. We designed several types of such novel predictors, all of which aimed at practically reducing the sparseness of the data matrix, while maintaining the most important clinical information as much as possible.

First, on the top of our predictor list are four novel predictors that are not present in the original tables. The `index_month_year_from_begin` and `age` predictors are calculated from the original columns `index_month_year` and `birth_yr`, respectively. They are kept as our top predictors to represent the dramatic difference in their distributions between the long-COVID vs. the short-COVID patients, shown below (Figure 2).

![Predictive features](img/RAAD_Figure2.jpg?raw=true)

The next two predictors, `earliest_to_covid_diag` and `latest_to_covid_diag`, were calculated using the original `days_to_covid_diag` column. `days_to_covid_diag` is a column that is present in the majority of data tables, including diagnoses, immunization, labs, medication and observations. It is a time series that shows a patient’s care history and response to treatment to COVID and related symptoms. However, it varies not only between the different patients, but also for each patient between different tables. In order to get the most representative collective information in all tables, we aggregated them into the two features listed above for each patient, to record the response speed and duration of COVID related medical interventions.

Second, as described in the Executive Summary, we have looked up the clinical meanings of the string-based categorical data presented in the diagnoses and medication tables, and simplified them to represent only the high-level categories that were clinically relevant. For example, all the different ICD-10-CM codes that start with `Z` are collapsed into the `ICD10_status_Z` to represent “a resolving disease, injury or chronic condition”. Similarly, `ICD10_status_R` represents “symptoms, signs and abnormal clinical and lab findings”, and `ICD10_status_J` for “condition in respiratory system”.

Third, for the lab tables, where over 2800 different tests with an extremely wide range of result values recorded, we tried to obtain a straightforward statistical representation of the data for each patient. For that purpose, we have manually checked the distribution of the top 50 lab tests based on their occurrences. Then considering whether extreme values are frequently observed, and how different the distribution is between the short vs. long-COVID patients, we chose one or multiple statistical metrics to represent each of the lab results. Statistical metrics we have finally selected includes the average, minimum, and maximum of the test result for each patient, as well as a value between 0 and 1 to indicate whether the results are within the normal range.

### How our work could influence further work
For pandemic and endemic, the timing of the diagnosis is an important factor to consider, as the virus or bacteria strains evolve over the time. From this data challenge, we discovered that most long COVID cases are after the Delta variant surfaced over the world. This result has highlighted the importance of our global surveillance efforts on tracking the evolution of infectious diseases and continuously suggesting new solutions accordingly.
 
EHR data contains errors and duplicates. During data preprocessing, researchers need to be mindful of extreme values (eg. >100 values for measurements with % as unit). We have also observed typos in strings of records, some corrections were applied. For certain variables that a patient reports him or herself, such as smoke history, inconsistency is commonly seen. We have dropped inconsistencies over several visits for the same patient. 


## Notes for someone rerunning this code
The following packages is required
```
pip install click
pip install fastparquet
pip install python-dateutil
pip install xgboost
pip install pyarrow
```
The code base require some preload files that are stored in a `preload` directory. The preload files needs to be generated once for a training data. If a new training data is available, a new preload files may be necessary.
 #### Run command for preload generation
 ```
 python preload.py -d /challenge/seeing-through-the-fog/data/train_data/ -o [Output Directory]
 ```
 The preload generation code takes the training data directory `(-d)` and an output directory `(-o)` to store the output preload file. 
**Note: The Top100_drug_with_annotation_category.txt preload file is manually generated. Plase see the Preprocessing section for description of this preload file generation process.**

#### Input Parameters

| Parameter             | Description                                |
|-----------------------|--------------------------------------------|
| --datapath (-d)        | Train or Test data directory path          |
| –-featuretabledir (-f) | Directory path to store the processed data |
| –-experimenttype (-t)  | Experimen type, i.e. Train, or Test        |
| –-experimentname (-n)  | Name of the experiment                     |
| –-help (-h)  | Show help message                     |

#### Run Command
`luminous` can run in `train` or `test` mode. The run command for `train` mode is 

```
python luminous.py -d /challenge/seeing-through-the-fog/data/train_data -f [Processed Data Store Directory] -t train -n [Experiment name]
```
The `train` mode get the training data, stores the processed raw data inside the directory for processed data. The data processing step creates a file with [experiment_name]_master_table_outer_merge.txt inside the directory for processed data. This file is used for data cleaning and feature selection. After feature selection is performed, the a file named `top_features.txt` is generated with feature names. The `top_feature.txt` file also stored inside the directory for processed data. The classification model that is tuned with training data is stored in a file named `trained_model.sav` inside the directory for processed data. The `top_features.txt` and `trained_model.sav` files is required to be located inside the directory for processed data before `test` mode is run.

The run command for `test` mode is

```
python luminous.py -d /challenge/seeing-through-the-fog/data/test_data -f [Processed Data Store Directory] -t test -n [Experiment name]
```
In the `test` mode, the data is processed and stored in the directory for processed data. **Note, the directory `(-f)` is the same as the one used during the `train` mode. This is required since the `top_features.txt` and `trained_model.sav` files are located in this directory and is required for test data prediction.** Once the prediction is performed, the output is stored in `predictions.parquet` file in the current directory.

:warning: **Note: The Processed Data Store Directory `(-f)` is created by the program. The user DO NOT need to create and provide an empty directory path, since it will throw an error.**

#### Example run command
Run `luminous` in `train` mode
```
python luminous.py -d /challenge/seeing-through-the-fog/data/train_data -f DataStore/ -t train -n training
```

Run `luminous` in `test` mode
```
python luminous.py -d /challenge/seeing-through-the-fog/data/test_data -f DataStore/ -t test -n testing
```
