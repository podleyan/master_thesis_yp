import pandas as pd
import numpy as np 
from calendar_data import getCalendarData
from weather import getWeatherData
from entsoe_data import getEntsoeData
from weather_forecast import getWeatherForecastData
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error


# Return three separate datasets with calendar, weather and electricity load data, 
# use for combining dataset for TFT and for RandomForest and XGBoost

def getDataBeforeMerge(country, fromDate, toDate):
    type = 'history'                                                    # Type of data for load of electricity load data
    entsoe_df = pd.DataFrame()                                          # Future electricity load dataframe

    # Get calendar data 
    calendar_df = getCalendarData(country, fromDate, toDate)
    calendar_df.set_index('date', inplace=True, drop=True)
    calendar_df.index = pd.to_datetime(calendar_df.index.values, utc=True)
    calendar_df.index.rename('timestamp',inplace=True)

    # Get electricity load data
    entsoe_df = getEntsoeData(country, fromDate, toDate, type)
    entsoe_df.index = pd.to_datetime(entsoe_df.index.values, utc=True)
    entsoe_df.index.rename('timestamp',inplace=True)
    entsoe_df['Actual Load'] = entsoe_df['Actual Load'].replace(0, np.nan)
    entsoe_df["Actual Load"] = entsoe_df['Actual Load'].interpolate()

    # Get weather load data
    weather_df =  getWeatherData(country, fromDate, toDate, 'hourly')
    weather_df.index = pd.to_datetime(weather_df.index.values, utc=True)
    weather_df.index.rename('timestamp',inplace=True)
    weather_df = weather_df.fillna(0)

    # Select only relevant weather columns 

    temp_columns = [f"{i:02d}_temp" for i in range(1, 11)]
    dwpt_columns = [f"{i:02d}_dwpt" for i in range(1, 11)]
    sun_column = ["08_tsun"]
    features = temp_columns + dwpt_columns + sun_column

    weather_df = select_relevant_weather_features(weather_df, features)
    
    return calendar_df, entsoe_df, weather_df


# Return combined data - final dataset for TFT
def getData(data_load, location, fromDate, toDate):
    if data_load:                                       # Load data from API or load data from local csv
        X = pd.DataFrame()

        # Get data for every country 

        for country in location: 
            entsoe_df = pd.DataFrame()
            forecast_df = pd.DataFrame()
            calendar_df, entsoe_df, weather_df = getDataBeforeMerge(country, fromDate, toDate)
            
            forecast_df['fct_temp'] = weather_df['01_temp']                                     # for training real data is used as a temperature forecast

            # Merge all the data together
            merged = entsoe_df.copy()
            merged = pd.merge(merged,calendar_df.loc[:, ["day", "weekday", "month", "holiday", "holiday_lag", "holiday_lead", "weekday_binary"]],how='left',left_index=True, right_index=True).ffill(limit=23)
            merged = pd.merge(merged,weather_df,how='left',left_index=True, right_index=True).ffill(limit=23)
            merged = pd.merge(merged,forecast_df,how='left',left_index=True, right_index=True).ffill(limit=23)
            merged = merged.drop_duplicates()
            
            merged['hour'] = merged.index.hour
            merged = merged.reset_index()
            merged.index.name = "time_idx"
            X = pd.concat([merged, X])

        # Transoform cyclical features to sin and cos representations

        X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24.0)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24.0)
        X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
        X['month_cos'] = np.cos(2 * np.pi * X['month']/12)

        X = X.reset_index()
    
    else: 
        # Load data from csv
        X = pd.read_csv('/Users/yanapodlesna/main/skool/master/X_countries.csv')
        X['timestamp']= pd.to_datetime(X['timestamp'])
 
    return X

def createLags(df, load_lag, get_country = True):

    df_lags = df.copy()
    if get_country: 
        for country in df['country'].unique():
            for i in load_lag:
                df_lags.loc[df['country'] == country, f'load_lag_{i}'] = df.loc[df['country'] == country, 'Actual Load'].shift(i)
    else: 
        for i in load_lag:
                df_lags[f'load_lag_{i}'] = df['Actual Load'].shift(i)

    df_lags.dropna(inplace=True)
    return df_lags


def split_in_time(df,date):
   # Creates train/test datasets by given date

    if isinstance(df.index, pd.Int64Index):
        # If index is integer, use the 'timestamp' column to split
        train = df.loc[df['timestamp'] <= date].copy()
        test = df.loc[df['timestamp'] > date].copy()
    else:
        # If index is not integer, use index to split
        train = df.loc[df.index <= date].copy()
        test = df.loc[df.index > date].copy()

    return train, test

# Return relevant weather features 

def select_relevant_weather_features(X, features):
    return X.loc[:, features]


def printMetrics(predicted_data):
    print('MAPE: ' + str(mean_absolute_percentage_error(predicted_data['Actual Load'], predicted_data['prediction'])))
    print('MSE: ' + str(mean_squared_error(predicted_data['Actual Load'], predicted_data['prediction'])))
    print('MAE: ' + str(mean_absolute_error(predicted_data['Actual Load'], predicted_data['prediction'])))
    print('R^2 score: '+ str(r2_score(predicted_data['Actual Load'], predicted_data['prediction'])))

def createCalendarFeatures(df): 
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['dayofmonth'] = df['timestamp'].dt.day
    df = df.set_index('timestamp')
    return df

def createFeatures(df_lags, df2=pd.DataFrame(),df3=pd.DataFrame(), label=None):
    """
    Creates time series features from datetime index \n
    df_lags = lags from entsoe \n
    df2 = weather \n
    df3 = calendar
    """

    df = df_lags.copy()
    
    df['date'] = df_lags.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    #df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    # df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour',
            'dayofweek',
            #'quarter',
            'month',
            'dayofmonth',
            'weekofyear'
           ]]
    

    # add lag entsoe data
    X = pd.merge(X,df_lags,how='left',left_index=True, right_index=True).ffill(limit=23)
    
    X = pd.merge(X,df3,how='left',left_index=True, right_index=True).ffill(limit=23)     
    X = X.drop_duplicates()
    
    # add weather data
    X = pd.merge(X,df2,how='left',left_index=True,right_index=True)
    X = X.reset_index() 
    #X = X.drop(0)
    X = X.set_index('timestamp', drop=True)

    if label:
        y = X[label]
        return X, y
    return X
