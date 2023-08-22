import pandas as pd

from data import getData, split_in_time, getDataBeforeMerge, printMetrics
import numpy as np 
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree

plt.style.use('fivethirtyeight')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

# Predict selected hour ahead with direct method with XGBoost and Random Forest (used for 6 hour ahead prediction)


# Create shifted features such as temperature forecast, weather and electricity load for specific hour
 
def model_data(entsoe_df, weather_df, calendar_df, shift):
    X = pd.DataFrame()
    forecast_df = pd.DataFrame()
    forecast_df['forecast_weather'] = weather_df['01_temp']
    for i in range(1,shift):
        forecast_df[f'forecast_weather_lag_{i+1}'] = forecast_df['forecast_weather'].shift(-i+1)
    
    weather_df = weather_df.shift(-shift)
    weather_df.columns = weather_df.columns + f'_lag_{shift}'


    entsoe_df[f'load_lag_{shift}'] = entsoe_df['Actual Load'].shift(-shift)
    entsoe_df['load_lag_24'] = entsoe_df['Actual Load'].shift(-24)
    entsoe_df['load_lag_48'] = entsoe_df["Actual Load"].shift(-48)
    entsoe_df['load_lag_168'] = entsoe_df["Actual Load"].shift(-168)
    y = pd.DataFrame()
    entsoe_df = entsoe_df.dropna()
    y['Actual Load'] = entsoe_df['Actual Load']
    entsoe_df = entsoe_df.drop(columns=['Actual Load'])
    merged = entsoe_df.copy()

    #print(entsoe_df)
    merged = pd.merge(merged,calendar_df.loc[:, ["weekday", "month", "holiday", "holiday_lag", "holiday_lead", "weekday_binary"]],how='left',left_index=True, right_index=True).ffill(limit=23)
    
    merged = pd.merge(merged,weather_df,how='left',left_index=True, right_index=True).ffill(limit=23)
    merged = pd.merge(merged,forecast_df,how='left',left_index=True, right_index=True).ffill(limit=23)

    ##merged = pd.merge(merged, forecast_df, how='left', left_index=True, right_index=True)
    merged = merged.drop_duplicates()


    merged['hour'] = merged.index.hour
    #X['dayofweek'] = X['timestamp'].dt.dayofweek
    #X['quarter'] = X['timestamp'].dt.quarter
    merged['month'] = merged.index.month
    merged['day'] = merged.index.day
    merged = merged.reset_index()
    merged.index.name = "time_idx"
    merged = merged.drop(columns=['country'])
    #X['weekofyear'] = X['timestamp'].dt.weekofyear
    X = pd.concat([merged, X])
    return X, y


#######################################################################################################################
# Program Functionality parameters

data_load = 1                                       # if want to load data or get it from csv         
hyperparameters_search = 0
#######################################################################################################################
# Create dataset

fromDate = 20210101                                 # start date for data load
toDate = 20230701                                   # end date for data load
location = {'CZ'}                                   # countries for data load (only CZ, HU or SK)

calendar_df, entsoe_df, weather_df = getDataBeforeMerge(location, fromDate, toDate)  # get data 
#######################################################################################################################
# Create features for model 

shift = 6                                            # data shift (hour of prediction)
X, y = model_data(entsoe_df, weather_df, calendar_df, shift)

#######################################################################################################################
# Split data on train/testing dataset

split_date = '2023-03-15'

X = X.set_index('timestamp')
X_train, X_test = split_in_time(X,split_date)
y_train, y_test = split_in_time(y,split_date)

data_test = pd.DataFrame()
entsoe_train, data_test = split_in_time(y, split_date)
#data_test = data_test.to_frame()

#######################################################################################################################
# Transform data
cat_attribs = ['hour','day','month', 'weekday_binary', 'holiday', 'holiday_lag', 'holiday_lead']

full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_attribs)], remainder='passthrough')
X_train = X_train.fillna(0)
encoder = full_pipeline.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)
#X_both = encoder.transform(X_both)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
#X_both = np.nan_to_num(X_both)

y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

#######################################################################################################################
# Random Forest Model

# Hyperparameters optimization with timeseries cross validation split 
 
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
# RF prediction and results

data_test['RF_prediction']  = rf.predict(X_test)
predicted_data = pd.DataFrame()
predicted_data['prediction'] = data_test['RF_prediction']
predicted_data['Actual Load'] = data_test['Actual Load']

printMetrics(predicted_data)

#######################################################################################################################
# XGBoost model

# Hyperparameters optimization
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
# Prediction with XGboost
y_pred = reg.predict(X_test)

# Add the predicted values to the DataFrame as a new column
data_test['XGB_prediction'] = y_pred
predicted_data['prediction'] = data_test['XGB_prediction']
printMetrics(predicted_data)

#######################################################################################################################
# Save data to csv 

data_test.to_csv(f'{shift}_model.csv')

#######################################################################################################################
# Create figure with prediction made with models

figure_date = '2023-01-15 00:00:00+00:00'

subset = ((pd.to_datetime(data_test.index) >= figure_date))
figure_df = data_test.loc[subset].copy()
figure_df_history = y[y.index.isin(pd.to_datetime(figure_df.index))].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=figure_df_history.index,y=figure_df_history['Actual Load'],name="Historical load"))
fig.add_trace(go.Scatter(x=figure_df.index,y=figure_df['RF_prediction'],name="RF"))
fig.add_trace(go.Scatter(x=figure_df.index,y=figure_df['XGB_prediction'],name="XGB"))
fig.update_layout(
    title="CZ Load",
    xaxis_title="Date",
    yaxis_title="Load"
)
fig.show()