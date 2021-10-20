from feature_enginering import df,workclass,race,country,occupation
import numpy as np
import pandas as pd
age_mean=df['age'].mean()
age_std=df['age'].std()
upper_bound=age_mean+3*age_std
lower_bound=age_mean-3*age_std
df = df[(df['age'] <= upper_bound) & (df['age'] >= lower_bound)]
df.drop(['workclass','fnlwgt','education','occupation',
    'relationship','race',
    'capital_gain','capital_loss','country'],axis = 1, inplace = True)
df.rename(columns={'age':'age_logarithmic'},inplace=True)

