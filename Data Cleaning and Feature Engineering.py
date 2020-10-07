# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:13:38 2020

@author: shubhs15
"""

# House Prices: Advanced Regression Techniques
# Predict sales prices and practice feature engineering, RFs, and gradient boosting

# =============================================================================
# Goal
# It is your job to predict the sales price for each house. For each Id in the test set, you must predict 
# the value of the SalePrice variable. 
# 
# Metric
# Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value
# and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses 
# and cheap houses will affect the result equally.)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/s-s.kumar/Desktop/Personal/KaggleRequirements/Project/house-prices-advanced-regression-techniques')
os.listdir()

houseprice_data = pd.read_csv('train.csv')
houseprice_data.head()
# [5 rows x 81 columns]

houseprice_data.shape
# Out[8]: (1460, 81)

pd.set_option('display.max_rows', None) # To display all the rows in console.
houseprice_data.isnull().sum()

houseprice_data.drop(['Id','Alley','PoolQC','Fence','MiscFeature'], axis=1, inplace = True)
houseprice_data.shape
# Out[24]: (1460, 76)

# Alternate method to find missing values
# Finding features which are having missing values in percent
nan_train_features = [features for features in houseprice_data.columns if houseprice_data[features].isnull().sum()>=1]
print(nan_train_features)

# Checking feature with missing values by plots
sns.heatmap(houseprice_data.isnull(),cbar=False,yticklabels=False)

# To find CATEGORICAL features having NAN values

house_category_nan = [features for features in houseprice_data.columns if houseprice_data[features].isnull().sum()>=1 
                      and houseprice_data[features].dtypes == 'O']

for feature in house_category_nan:
    print(feature, ':', np.round(houseprice_data[feature].isnull().mean(),3), '% missing values')

# To find NUMERICAL features having NAN values

house_numerical_nan = [features for features in houseprice_data.columns if houseprice_data[features].isnull().sum()>=1 
                      and houseprice_data[features].dtypes != 'O']

for feature in house_numerical_nan:
    print(feature, ':', np.round(houseprice_data[feature].isnull().mean(),3), '% missing values')
    
# Replacing missing values in categorical features with a mode
for feature in house_category_nan:
    mode_value = houseprice_data[feature].mode()[0]
    houseprice_data[feature].fillna(mode_value,inplace=True)
    
houseprice_data[house_category_nan].isnull().sum()

# Replacing missing values in numerical features with a median value
for feature in house_numerical_nan:
    median_value = houseprice_data[feature].median()
    houseprice_data[feature+'nan'] = np.where(houseprice_data[feature].isnull(),1,0)
    houseprice_data[feature].fillna(median_value,inplace=True)
    
houseprice_data[house_numerical_nan].isnull().sum()

houseprice_data.columns
houseprice_data.drop(['LotFrontagenan', 'MasVnrAreanan','GarageYrBltnan'], axis=1, inplace=True)

main_houseprice_data = houseprice_data.copy()

# TEST DATASET

os.chdir('C:/Users/s-s.kumar/Desktop/Personal/KaggleRequirements/Project/house-prices-advanced-regression-techniques')
os.listdir()

housetest_data = pd.read_csv('test.csv')
housetest_data.head()
# [5 rows x 81 columns]

housetest_data.shape
# Out[8]: (1460, 80)

pd.set_option('display.max_rows', None) # To display all the rows in console.
housetest_data.isnull().sum()

housetest_data.drop(['Id','Alley','PoolQC','Fence','MiscFeature'], axis=1, inplace = True)
housetest_data.shape
# Out[34]: (1459, 75)

# To find CATEGORICAL features having NAN values in test dataset

housetest_category_nan = [features for features in housetest_data.columns if housetest_data[features].isnull().sum()>=1 
                      and housetest_data[features].dtypes == 'O']

for feature in housetest_category_nan:
    print(feature, ':', np.round(housetest_data[feature].isnull().mean(),3), '% missing values')

# Replacing missing values in categorical features with a mode
for feature in housetest_category_nan:
    modetest_value = housetest_data[feature].mode()[0]
    housetest_data[feature].fillna(mode_value,inplace=True)
    
housetest_data[housetest_category_nan].isnull().sum()

# To find NUMERICAL features having NAN values

housetest_numerical_nan = [features for features in housetest_data.columns if housetest_data[features].isnull().sum()>=1 
                      and housetest_data[features].dtypes != 'O']

for feature in housetest_numerical_nan:
    print(feature, ':', np.round(housetest_data[feature].isnull().mean(),3), '% missing values')

# Replacing missing values in numerical features with a median value
for feature in housetest_numerical_nan:
    mediantest_value = housetest_data[feature].median()
    housetest_data[feature+'nan'] = np.where(housetest_data[feature].isnull(),1,0)
    housetest_data[feature].fillna(mediantest_value,inplace=True)
    
housetest_data[housetest_numerical_nan].isnull().sum()

housetest_data.columns
housetest_data.drop(['LotFrontagenan', 'MasVnrAreanan', 'BsmtFinSF1nan','BsmtFinSF2nan', 'BsmtUnfSFnan', 'TotalBsmtSFnan', 
                     'BsmtFullBathnan','BsmtHalfBathnan', 'GarageYrBltnan', 'GarageCarsnan', 'GarageAreanan'], axis=1, inplace=True)

housetest_data.to_csv('cleanedtestdata.csv',index=False)

testhouse_data = pd.read_csv('cleanedtestdata.csv')

# Concating the train and test house dataset
final_housedata = pd.concat([main_houseprice_data,testhouse_data], axis=0)
final_housedata.shape
# Out[53]: (2919, 76)

final_housedata.isnull().sum()

# Changing the Cetegorical Features to Numeric Feature (Label Encoding)
from sklearn.preprocessing import LabelEncoder

for i in range(0,len(final_housedata.columns)):
    col_name = final_housedata.columns[i]
    if(isinstance(final_housedata.iloc[0,i],str)):
        encoder = LabelEncoder()
        encoder.fit(final_housedata[col_name])
        final_housedata[col_name] = encoder.transform(final_housedata[col_name])


# Splitting the final concatenated dataset to respective train and test house dataset
housefinal_train = final_housedata.iloc[:1460,:]
housefinal_train.shape
# Out[64]: (1460, 76)

housefinal_test = final_housedata.iloc[1460:,:]
housefinal_test.shape
# Out[65]: (1459, 76)

housefinal_test.drop(['SalePrice'], axis=1, inplace=True)
housefinal_test.shape

# TASKS COMPLETED 
# A. Found out the features with nan variables (nan_train_features)
# B. Deleted features which were having more than 80% of missing values.
# C. Deleted ID column from the dataset
# D. Found out the categorical features with nan variables and their percent (house_category_nan, housetest_category_nan)
# E. Found out the numerical features with nan variables and their percent (house_numerical_nan, housetest_numerical_nan)
# F. Replaced categorical features missing value with mode value
# G. Replaced numerical features missing value with median value
# H. Concatenated the train and test data in a single dataset
# I. Converted all categorical features to numerical features using LabelEncoding
# J. Split the concatenated final dataset back to train and test house dataset