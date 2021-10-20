from connect_database import df
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

df = df.iloc[:, 1:]
df.replace(to_replace=df['marital_status'].unique(), \
           value=['single', 'married', 'single', 'single', 'single', \
                  'married', 'single'], inplace=True)
df.replace(to_replace=['single', 'married'], value=[0, 1], inplace=True)
df.replace(to_replace=[' <=50K', ' >50K'], value=[0, 1], inplace=True)

gain_or_loss = np.zeros(len(df))
gain_index = df[df['capital_gain'] != 0].index
loss_index = df[df['capital_loss'] != 0].index
for index in gain_index:
    gain_or_loss[index] = 1
for index in loss_index:
    gain_or_loss[index] = -1

df['gain/loss'] = gain_or_loss.astype(int)
df.replace(to_replace=[' Female', ' Male'], value=[0, 1], inplace=True)
df_education_labels = df.groupby(by='education').describe()['education_num']['mean'].sort_values().reset_index()
df.rename(columns={'education_num': 'education_rank'}, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=' ?', strategy='most_frequent')
workclass_imputed = imputer.fit_transform(df[['workclass']])
df['workclass'] = workclass_imputed
df['workclass'].value_counts(normalize=True) * 100
arr_others = df['workclass'].value_counts(normalize=True)[df['workclass'].value_counts(normalize=True) * 100 < 2].index
df.replace(to_replace=arr_others, value=['others'] * len(arr_others), inplace=True)
workclass = pd.get_dummies(df[['workclass']], drop_first=True)
workclass.columns = ['Local_gov', 'Private', 'Self_emp_inc', 'Self_emp_not_in_inc', 'State_gov', 'others']
race = pd.get_dummies(df[['race']], drop_first=True)
race.columns = ['Asian_Pac_Islander', 'Black', 'Other', 'White']
imputer = SimpleImputer(missing_values=' ?', strategy='most_frequent')
occupation_imputed = imputer.fit_transform(df[['occupation']])
df['occupation'] = occupation_imputed
occupation = pd.get_dummies(df[['occupation']], drop_first=True)
occupation.columns = ['Armed_Forces', 'Craft_repair', 'Exec_managerial', 'Farming_fishing', 'Handlers_cleaners',
                      'Machine_op_inspct', 'Other_service', 'Priv_house_serv', 'Prof_specialty', 'Protective_serv',
                      'Sales', 'Tech_support', 'Transport_moving']
imputer = SimpleImputer(missing_values=' ?', strategy='most_frequent')
country_imputed = imputer.fit_transform(df[['country']])
df['country'] = country_imputed
percentage_threshold = 0.3
arr_others = df['country'].value_counts()[df['country'].value_counts(normalize=True) * 100 < percentage_threshold].index
df['country'].replace(to_replace=arr_others, value=['others'] * len(arr_others), inplace=True)
country = pd.get_dummies(df[['country']], drop_first=True)
country.columns = ['El Salvador', 'Germany', 'India', 'Mexico', 'Philippines', 'Puerto Rico', 'United States', 'others']
