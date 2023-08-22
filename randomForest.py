
import pandas as pd
from data import getData, createLags, split_in_time, printMetrics, getDataBeforeMerge, createLagsY, createFeatures
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
#######################################################################################################################
# Program Functionality parameters

data_load = 1 
hyperparameters_search = 0 

#######################################################################################################################
# Data load 

fromDate = 20210101                                 # start date for data load
toDate = 20230701                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only 1 from CZ, HU, SK)
load_lag = [1, 24, 48, 168]                         # lags we want to create

calendar_df, entsoe_df, weather_df = getDataBeforeMerge(location, fromDate, toDate)  # get data 
print(entsoe_df)

data_lags, y = createLagsY(pd.DataFrame(entsoe_df['Actual Load']), load_lag=load_lag)

X = createFeatures(data_lags, df2 = weather_df, df3 = calendar_df.loc[:, ["weekday", "month", "holiday"]])
#X = create_features(data_lags, df2 = weather_df.iloc[:,[0]], df3 = calendar_df.iloc[:,[0,1,2]], country=country)

#######################################################################################################################
# Train test split

split_date = '2023-03-15'

X_train, X_test = split_in_time(X,split_date)
y_train, y_test = split_in_time(y,split_date)


data_test = pd.DataFrame()
entsoe_train, data_test = split_in_time(entsoe_df['Actual Load'], split_date)

#######################################################################################################################
# Transform data
cat_attribs = ['hour','dayofweek','month_x','weekofyear']

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

count = 0

for row in range(len(X[(X.index >= start_time) & (X.index < end_time)])):
# for row in tqdm(range(7,10)):
    X = createFeatures(entsoe_df, df2 = weather_df, df3 = calendar_df.loc[:,["month", "holiday", "weekday"]])

    print('Row number ', row, ' out of ', len(X[X.index >= start_time]))
    
    fromPredict = X[X.index >= start_time].iloc[row]
    fromPredictionTime = fromPredict.name
    
    hour = fromPredictionTime.hour
    
    for i in range(1, 37): 
        count=count+1
        
        # create new lags
        data, y = createLagsY(X, load_lag=load_lag)
        df = X.copy()
        df.drop(columns=['Actual Load'],inplace=True)
        df = df.merge(data,how='left',left_index=True,right_index=True)

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
        X.loc[X[X.index >= start_time].index[row + i],'Actual Load'] = prediction
        
        df_RF.loc[count - 1, 'time'] = fromPredictionTime
        df_RF.loc[count - 1, 'prediction'] = prediction[0]
        df_RF.loc[count - 1, 'forecasted_time'] = timeOfPrediction
        #print('Prediction from ', fromPredictionTime, ' load predicted ', prediction[0][0], ' time of load', timeOfPrediction)

print('done') 

df_RF.to_csv('RF.csv')
rf_results = pd.merge(df_RF, X["Actual Load"], how='left', left_on='forecasted_time', right_on = 'timestamp')
#######################################################################################################################
# Results 

printMetrics(rf_results)  

