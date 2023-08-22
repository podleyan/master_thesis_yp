import pandas as pd
from data import getData, createLags, split_in_time, printMetrics, getDataBeforeMerge
import xgboost as xgb
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Predicting 36 hour ahead electricity load in Czech Republic with XGBoost in reccurent way

#######################################################################################################################
# Program Functionality parameters

data_load = 1 
hyperparameters_search = 0 
prediction_length = 36

#######################################################################################################################
# Data load 
# In the thesis dataset was used generated from date = 2022-01-01 to date = '2023-07-01' with '2023-03-15 as splitting date' 


fromDate = 20230101                                 # start date for data load
toDate = 20230323                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only 1 from CZ, HU, SK)
load_lag = [1, 24, 48, 168]                         # lags we want to create


X_without_lags = getData(data_load, location, fromDate, toDate)  # get data 
X_without_lags = X_without_lags.drop(columns={'time_idx', 'country', 'month_sin', 'month_cos', 'hour_cos', 'hour_sin'})
X_without_lags = X_without_lags.set_index('timestamp')

# For XGBoost training with only one weather feature was the best, uncomment next row to set up relevant features

#X_without_lags = select_relevant_weather_features(X_without_lags, '04_temp')
X = createLags(X_without_lags, load_lag=load_lag, get_country = False)

y = X['Actual Load']
X = X.drop(columns={'Actual Load'})

#######################################################################################################################
# Train test split

split_date = '2023-03-15'

X_train, X_test = split_in_time(X,split_date)
y_train, y_test = split_in_time(y,split_date)


data_test = pd.DataFrame()
entsoe_train, data_test = split_in_time(y, split_date)

#######################################################################################################################
# Transform data
cat_attribs = ['day', 'weekday', 'hour', 'month']

full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_attribs)], remainder='passthrough')
encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

#######################################################################################################################
# Hyperparameter optimization and model training
# Example from the thesis has this hyperparameters: 
# reg = xgb.XGBRegressor(n_estimators=1000, max_depth=25, learning_rate=0.1, reg_alpha=0.2, reg_lambda=1.2, objective = 'reg:squarederror', min_child_weight = 5)


if hyperparameters_search: 
    tscv = TimeSeriesSplit(n_splits = 3)

    model = xgb.XGBRegressor()
    parameters = {
        'max_depth': [15, 25, 50],
        'n_estimators': [1000, 1500, 2000],
        'learning_rate': [0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.2, 0.5],
        'reg_lambda': [1, 1.2, 1.5],
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

    reg = xgb.XGBRegressor(n_estimators=1000, max_depth=25, learning_rate=0.1, reg_alpha=0.2, reg_lambda=1.2, objective = 'reg:squarederror')
    reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)

#######################################################################################################################
# Predicting 36 hours ahead 

#######################################################################################################################
# Predicting 36 hours ahead 

start_time = split_date
end_time = X.index.max()
df_xgb =pd.DataFrame(columns={"time", "prediction", "forecasted_time"})
#X_without_lags = X_without_lags.drop(columns = {'Actual Load'})
count = 0
end_row = len((X[X.index >= start_time]))-prediction_length

for row in range(len(X[(X.index >= start_time) & (X.index < end_time)])-prediction_length+1):
# for row in tqdm(range(7,10)):
    X_temp = X_without_lags.copy()

    print('Row number ', row, ' out of ', end_row-1)
    
    fromPredict = X_temp[X_temp.index >= start_time].iloc[row]
    fromPredictionTime = fromPredict.name
    
    hour = fromPredictionTime.hour
    
    for i in range(1, prediction_length+1): 
        count=count+1

        # create new lags
        df = createLags(X_temp, load_lag=load_lag, get_country= False)
        df = df.drop(columns = {'Actual Load'})

        # prediction data
    
        rowToPredict = df[df.index >= start_time].iloc[row + i]

        timeOfPrediction = rowToPredict.name

        rowHelp = df[df.index >= start_time].copy()
        rowToPredict = rowHelp.iloc[[row + i]]

        #transform data for xgboost
        rowToPredictEncode = encoder.transform(rowToPredict)
        
        # make prediction
        prediction = reg.predict(rowToPredictEncode)

        # update X df and store prediction
        X_temp.loc[X_temp[X_temp.index >= start_time].index[row + i],'Actual Load'] = prediction

        df_xgb.loc[count - 1, 'time'] = fromPredictionTime
        df_xgb.loc[count - 1, 'prediction'] = prediction[0]
        df_xgb.loc[count - 1, 'forecasted_time'] = timeOfPrediction
        #print('Prediction from ', fromPredictionTime, ' load predicted ', prediction[0][0], ' time of load', timeOfPrediction)
print('Prediction is done') 


df_xgb.to_csv('XGB.csv')                       # Save results to csv
df_xgb['forecasted_time'] = pd.to_datetime(df_xgb['forecasted_time'], utc=True)
xgb_results = pd.merge(df_xgb, y, how='left', left_on='forecasted_time', right_on='timestamp')


#######################################################################################################################
# Results 

printMetrics(xgb_results)  

