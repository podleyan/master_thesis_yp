from data import getData
# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_load = 1      
fromDate = 20210101                                 # start date for data load
toDate = 20230701                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only CZ, HU or SK)
load_lag = [24, 48, 168]                            # lags we want to create

X = getData(data_load, location, fromDate, toDate)  # get data 


df = X.loc[X['country'] == 'CZ']
df = df.drop(['country', 'timestamp'], axis=1)

#######################################################################################################################
# Correlation Analysis

corr_matrix = df.corr()
abs(corr_matrix["Actual Load"]).sort_values(ascending=False).to_csv('features.csv')

plt.figure(figsize=(5,50))  # adjust size to fit
sns.heatmap(abs(corr_matrix[["Actual Load"]]).sort_values(by="Actual Load", ascending=False), annot=True, cmap='coolwarm')



#######################################################################################################################
# Feature importance

# Specify the feature and target variables
X1 = df[['weekday','month','holiday','01_temp','01_dwpt','01_rhum','01_prcp','01_snow','01_wdir','01_wspd','01_wpgt','01_pres','01_tsun','01_coco','02_temp','02_dwpt',
       '02_rhum','02_prcp','02_snow','02_wdir','02_wspd','02_wpgt','02_pres','02_tsun','02_coco','03_temp','03_dwpt','03_rhum','03_prcp','03_snow','03_wdir','03_wspd','03_wpgt','03_pres',
       '03_tsun','03_coco','04_temp','04_dwpt','04_rhum','04_prcp','04_snow','04_wdir','04_wspd','04_wpgt','04_pres','04_tsun','04_coco','05_temp','05_dwpt','05_rhum','05_prcp','05_snow',
       '05_wdir','05_wspd','05_wpgt','05_pres','05_tsun','05_coco','06_temp','06_dwpt','06_rhum','06_prcp','06_snow','06_wdir','06_wspd','06_wpgt','06_pres','06_tsun','06_coco','07_temp',
       '07_dwpt','07_rhum','07_prcp','07_snow','07_wdir','07_wspd','07_wpgt','07_pres','07_tsun','07_coco','08_temp','08_dwpt','08_rhum','08_prcp','08_snow','08_wdir','08_wspd','08_wpgt',
       '08_pres','08_tsun','08_coco','09_temp','09_dwpt','09_rhum','09_prcp','09_snow','09_wdir','09_wspd','09_wpgt','09_pres','09_tsun','09_coco','10_temp','10_dwpt','10_rhum','10_prcp',
       '10_snow','10_wdir','10_wspd','10_wpgt','10_pres','10_tsun','10_coco','hour','day', "holiday_lag", "holiday_lead", "weekday_binary"
]]
y = df['Actual Load']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importances = pd.DataFrame({'feature': X1.columns, 'importance': importances})

# Print out the feature and their importances
print(feature_importances.sort_values('importance', ascending=False))
feature_importances.sort_values('importance', ascending=False).to_csv('features.csv')