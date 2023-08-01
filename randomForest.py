
import pandas as pd
from calendar_data import getCalendarData
from weather import getWeatherData
from entsoe_data import getEntsoeData
from data import getData, createLags, split_in_time, printMetrics

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm


data_load = 1 
hyperparameters_search = 0 
fromDate = 20210101                                 # start date for data load
toDate = 20230701                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only 1 from CZ, HU, SK)
load_lag = [1, 24, 48, 168]                         # lags we want to create

X = getData(data_load, location, fromDate, toDate)  # get data 

data = X.drop(columns = {'Actual Load'})

X = createLags(X, load_lag)

y = pd.DataFrame()
X = X.set_index('timestamp')
y['Actual Load'] = X['Actual Load']
y_temp = y 
X = X.reset_index()
y = y.reset_index()


X = X.drop(columns = {'Actual Load', 'country'})

split_date = '2023-03-15'

X_train, X_test = split_in_time(X,split_date)
y_train, y_test = split_in_time(y,split_date)

y_train.set_index('timestamp')
y_test.set_index('timestamp')

X_train = X_train.drop(columns='timestamp')
X_test = X_test.drop(columns='timestamp')

# For y, just keep the 'Actual Load' column
y_train = y_train['Actual Load'].values
y_test = y_test['Actual Load'].values

cat_attribs = ['hour','month','weekday', 'holiday']

full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_attribs)], remainder='passthrough')
X_train = X_train.fillna(0)
encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

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

#rf = RandomForestRegressor(random_state=42,n_jobs=4,criterion="mse",max_depth=30,max_features='auto',n_estimators=500,warm_start=True)
rf = RandomForestRegressor(random_state=42,n_jobs=4,criterion ="squared_error",max_depth=20,max_features='auto',n_estimators=500,warm_start=True)
rf.fit(X_train, y_train)



start_time = split_date
end_time = X.index.max()
df_RF =pd.DataFrame(columns={"time", "forecasted_load", "forecasted_time"})


count = 0
for row in range(len(X[(X['timestamp'] >= start_time) & (X.index < end_time)])):
# for row in tqdm(range(7,10)):
    load_df = y_temp 
    fromPredict = X[X.index >= start_time].iloc[row]
    fromPredictionTime = fromPredict.name
    
    hour = fromPredictionTime.hour
    
    for i in range(1, 37): 
        count=count+1
        laged_load_df = createLags(load_df, load_lag)
        laged_load_df = laged_load_df.drop(columns={'Actual Load'})

        df = data.merge(laged_load_df, how = 'inner')
        # prediction data
        rowToPredict = data[data['timestamp'] >= start_time].iloc[row + i]
        timeOfPrediction = rowToPredict.name

        rowHelp = data[data['timestamp'] >= start_time].copy()
        rowToPredict = rowHelp.iloc[[row + i]]

        #transform data for xgboost
        rowToPredictEncode = encoder.transform(rowToPredict)
        
        # make prediction
        prediction = rf.predict(rowToPredictEncode)
        
        # update X df and store prediction
        load_df.loc[load_df[load_df.index >= start_time].index[row],'Actual Load'] = prediction
        
        df_RF.loc[count - 1, 'time'] = fromPredictionTime
        df_RF.loc[count - 1, 'forecasted_load'] = prediction[0]
        df_RF.loc[count - 1, 'forecasted_time'] = timeOfPrediction
        #print('Prediction from ', fromPredictionTime, ' load predicted ', prediction[0][0], ' time of load', timeOfPrediction)

print('done')    


