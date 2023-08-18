import pandas as pd
import numpy as np 
from calendar_data import getCalendarData
from weather import getWeatherData
from entsoe_data import getEntsoeData
from forecast_data import getWeatherForecast

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error

def getDataBeforeMerge(location, fromDate, toDate):
    type = 'history'
    for country in location: 
            entsoe_df = pd.DataFrame()
            forecast_df = pd.DataFrame()
        #if country == 'CZ':
        #    forecast_df = pd.read_csv("cz_forecast.csv", sep = ';')
        #if country == 'SK':
        #    forecast_df = pd.read_csv("sk_forecast.csv", sep = ';')
        #if country == 'HU':
        #    forecast_df = pd.read_csv("hu_forecast.csv", sep = ';')
        #forecast_df = forecast_df.set_index('time')
        #forecast_df.index = pd.to_datetime(forecast_df.index, utc=True)

    calendar_df = getCalendarData(country, fromDate, toDate)
    calendar_df.set_index('date', inplace=True, drop=True)
    calendar_df.index = pd.to_datetime(calendar_df.index.values, utc=True)
    calendar_df.index.rename('timestamp',inplace=True)


    entsoe_df = getEntsoeData(country, fromDate, toDate, type)
    entsoe_df.index = pd.to_datetime(entsoe_df.index.values, utc=True)
    entsoe_df.index.rename('timestamp',inplace=True)
    entsoe_df['Actual Load'] = entsoe_df['Actual Load'].replace(0, np.nan)
    entsoe_df["Actual Load"] = entsoe_df['Actual Load'].interpolate()



    weather_df =  getWeatherData(country, fromDate, toDate, 'hourly')
    weather_df.index = pd.to_datetime(weather_df.index.values, utc=True)
    weather_df.index.rename('timestamp',inplace=True)
    weather_df = weather_df.fillna(0)
    return calendar_df, entsoe_df, weather_df


def getData(data_load, location, fromDate, toDate):
    if data_load:
        X = pd.DataFrame()
        for country in location: 
            entsoe_df = pd.DataFrame()
            forecast_df = pd.DataFrame()
        calendar_df, entsoe_df, weather_df = getDataBeforeMerge(location, fromDate, toDate)
    ##forecast_df = pd.merge(weather_df['01_temp'], forecast_df, how='left', left_index=True, right_index=True)
    ##print(forecast_df)
    # Create a new column 'merged_temperature' and initialize it with NaN values
    ##forecast_df['merged_temperature'] = pd.Series(dtype=float)

    # Set the temperature values based on the timestamp condition
    ##forecast_df.loc[forecast_df.index >= '2023-03-15 00:00', 'fct_temp'] = forecast_df['temperature']
    ##forecast_df.loc[forecast_df.index < '2023-03-15 00:00', 'fct_temp'] = forecast_df['01_temp']

    # Drop the redundant temperature columns
    ##forecast_df = forecast_df.drop(['01_temp', 'temperature'], axis=1)

# Print the resulting merged DataFrame
    ##print(forecast_df)

        merged = entsoe_df.copy()
        merged = pd.merge(merged,calendar_df.loc[:, ["weekday", "month", "holiday", "holiday_lag", "holiday_lead", "weekday_binary"]],how='left',left_index=True, right_index=True).ffill(limit=23)
        merged = pd.merge(merged,weather_df,how='left',left_index=True, right_index=True).ffill(limit=23)
    ##merged = pd.merge(merged, forecast_df, how='left', left_index=True, right_index=True)
        merged = merged.drop_duplicates()


        #merged['date'] = merged.index.date
        merged['hour'] = merged.index.hour
    #X['dayofweek'] = X['timestamp'].dt.dayofweek
    #X['quarter'] = X['timestamp'].dt.quarter
        merged['month'] = merged.index.month
        merged['day'] = merged.index.day
        merged = merged.reset_index()
        merged.index.name = "time_idx"
        X = pd.concat([merged, X])
        X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24.0)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24.0)
        X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
        X['month_cos'] = np.cos(2 * np.pi * X['month']/12)
        X['Actual Load'] = X['Actual Load'].replace(0, np.nan)
        X['Actual Load'] = X['Actual Load'].interpolate()
    else: 
        X = pd.read_csv('/Users/yanapodlesna/main/skool/master/X_countries.csv')
        X['timestamp']= pd.to_datetime(X['timestamp'])
 
    return X

def createLags(df, load_lag):
    
    for country in df['country'].unique():

        #y = pd.DataFrame(df['Actual Load'])
        df_lags = df
        for i in load_lag:
            df_lags.loc[df['country'] == country, f'load_lag_{i}'] = df.loc[df['country'] == country, 'Actual Load'].shift(i)
        df_lags = df_lags[df_lags['timestamp'] > '2021-01-08']

    return df_lags
#, y
def createLagsY(df, load_lag):
    y = pd.DataFrame(df['Actual Load'])
    #y = pd.DataFrame(df['{}_forecast'.format(country)])
    df_lags = pd.DataFrame()

    # create lag variables by hours for load
    for i in load_lag:
        df_lags['load_lag_{}'.format(i)] = df['Actual Load'].shift(i)

    return df_lags, y

def split_in_time(df,date):
    """
    Creates train/test datasets by given date
    """

    # split 
    train = df.loc[df.index <= date].copy()
    test = df.loc[df.index> date].copy()

    return train, test

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