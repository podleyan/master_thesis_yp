
import pandas as pd
from data import getData, createLags, split_in_time, printMetrics, getDataBeforeMerge
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# Predicting 36 hour ahead electricity load in Czech Republic with Random Forest in reccurent way

#######################################################################################################################
# Program Functionality parameters

data_load = 1 
hyperparameters_search = 0 
prediction_length = 36

#######################################################################################################################
# Data load 
# In the thesis dataset was used generated from date = 2022-01-01 to date = '2023-07-01' with '2023-03-15 as splitting date' 


fromDate = 20230101                                 # start date for data load
toDate = 20230320                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only 1 from CZ, HU, SK)
load_lag = [1, 24, 48, 168]                         # lags we want to create


X_without_lags = getData(data_load, location, fromDate, toDate)  # get data 
X_without_lags = X_without_lags.drop(columns={'time_idx', 'country', 'month_sin', 'month_cos', 'hour_cos', 'hour_sin'})
X_without_lags = X_without_lags.set_index('timestamp')

X = createLags(X_without_lags, load_lag=load_lag, get_country = False)

y = X['Actual Load']
X = X.drop(columns={'Actual Load'})

print(y)

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

# In thesis this set of hyperparameters was used: 
# rf = RandomForestRegressor(random_state=42,n_jobs=4,criterion ="squared_error",max_depth=20,max_features='auto',n_estimators=500,warm_start=True)

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


#######################################################################################################################
# Predicting 36 hours ahead 

start_time = split_date
end_time = X.index.max()
df_RF =pd.DataFrame(columns={"time", "prediction", "forecasted_time"})
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
        prediction = rf.predict(rowToPredictEncode)

        # update X df and store prediction
        X_temp.loc[X_temp[X_temp.index >= start_time].index[row + i],'Actual Load'] = prediction

        df_RF.loc[count - 1, 'time'] = fromPredictionTime
        df_RF.loc[count - 1, 'prediction'] = prediction[0]
        df_RF.loc[count - 1, 'forecasted_time'] = timeOfPrediction
        #print('Prediction from ', fromPredictionTime, ' load predicted ', prediction[0][0], ' time of load', timeOfPrediction)
print('Prediction is done') 


df_RF.to_csv('RF.csv')                       # Save results to csv
df_RF['forecasted_time'] = pd.to_datetime(df_RF['forecasted_time'], utc=True)
rf_results = pd.merge(df_RF, y, how='left', left_on='forecasted_time', right_on='timestamp')
#######################################################################################################################
# Results 

printMetrics(rf_results)  

