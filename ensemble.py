import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from data import createCalendarFeatures, split_in_time, printMetrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

# Predictions of ensemble model trained on predicted values of 
# Random Forest, XGBoost and TFT models and calendar data and electricity load for specific hour
# Trained data containes prediction for dates '2022-08-26' to '2023-03-15' and tested on data from '2023-03-15' to '2023-07-01'



#######################################################################################################################
# Program Functionality parameters
hour = 2                                                # hour to predict
hyperparameters_search = 0                              # hyperparameter search

#######################################################################################################################
# Data Load and Transform
tft = pd.read_csv('/Users/yanapodlesna/main/skool/master/predicted_data_6h_2.csv')
tft['time_difference'] = (pd.to_datetime(tft['forecasted_timestamp']) - pd.to_datetime(tft['timestamp'])).dt.total_seconds() / 3600 
tft =tft.drop(columns={'time_idx', 'Unnamed: 0', "Actual Load", 'timestamp'})
tft =tft.rename(columns={'prediction': 'TFT_prediction', 'forecasted_timestamp': 'timestamp'})

model = pd.read_csv(f'/Users/yanapodlesna/main/skool/master/{hour}_model.csv')

tft = tft[tft['time_difference'] == hour]
tft = tft.drop(columns={'time_difference'})

merged = pd.merge(tft, model, on = 'timestamp', how = 'inner')
merged = createCalendarFeatures(merged)

y = merged['Actual Load']
X = merged.drop(columns={'Actual Load'})

#######################################################################################################################
# Create train and test datasets

split_date = '2023-05-15'

X_train, X_test = split_in_time(X,split_date)
y_train, y_test = split_in_time(y,split_date)

data_test = pd.DataFrame()
entsoe_train, data_test = split_in_time(y, split_date)
#data_test = data_test.to_frame()

#######################################################################################################################
# Transform data

cat_attribs = ['hour','dayofweek','month', 'dayofyear', 'dayofmonth']

full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_attribs)], remainder='passthrough')
X_train = X_train.fillna(0)
encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

#######################################################################################################################
# Hyperparameters search and model training 

if hyperparameters_search: 
    tscv = TimeSeriesSplit(n_splits = 3)

    model = RandomForestRegressor()
    parameters = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_depth' : [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }



    #gsearch = GridSearchCV(estimator=model, cv=tscv,
    #                         param_grid=parameters,
    #                        verbose=10)

    gsearch = RandomizedSearchCV(estimator=model, cv=tscv,
                             param_distributions=parameters,
                             n_iter=60,
                             verbose=10)

    gsearch.fit(X_train, y_train)
else:
    rf = RandomForestRegressor(random_state=42,n_jobs=4,criterion ="squared_error",max_depth=20,max_features='auto',n_estimators=500,warm_start=True)
    rf.fit(X_train, y_train)


data_test= data_test.reset_index()
data_test['prediction'] = rf.predict(X_test)

#######################################################################################################################
# Results 

printMetrics(data_test)

data_test.to_csv(f'ensembe{hour}.csv')