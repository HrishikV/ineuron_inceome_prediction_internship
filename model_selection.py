from  featur_selection import df,race,occupation,workclass,country
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns
df1=df.copy()
salary=df1['salary'].reset_index(drop=True)
df1=df1.drop(['salary'],axis=1)
def concat_dataframes(data):
    dataframe = pd.concat([data, workclass.iloc[data.index, :],                           race.iloc[data.index , :],                           occupation.iloc[data.index, :],                           country.iloc[data.index, :]], axis = 1)

    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

df1= concat_dataframes(df1)
features=['age_logarthmic','hours_per_week']
scaler = ColumnTransformer(transformers = [('scale_num_features', StandardScaler(), features)], remainder='passthrough')

models = [LogisticRegression(), SVC(), AdaBoostClassifier(), RandomForestClassifier(), XGBClassifier(),DecisionTreeClassifier(), KNeighborsClassifier(), CatBoostClassifier()]
model_labels = ['LogisticReg.','SVC','AdaBoost','RandomForest','Xgboost','DecisionTree','KNN', 'CatBoost']
mean_validation_f1_scores = []

for model in models:

  data_pipeline = Pipeline(steps = [
                                    ('scaler', scaler),
                                    ('resample', SMOTETomek()),
                                    ('model', model)
  ])
  mean_validation_f1 = float(cross_val_score(data_pipeline, df1, salary, cv=KFold(n_splits=10), scoring='f1',n_jobs=-1).mean())
  mean_validation_f1_scores.append(mean_validation_f1)
print(mean_validation_f1_scores)
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15,8))

sns.set_style('dark')
sns.barplot(y = model_labels ,x = mean_validation_f1_scores, ax=axes[0])
axes[0].grid(True, color='k')

sns.set_style('whitegrid')
sns.lineplot(x = model_labels, y = mean_validation_f1_scores)
axes[1].grid(True, color='k')
fig.show()