import pandas as pd

#check why weather has nans 
def model_data(entsoe_df, weather_df, calendar_df, shift):
    X = pd.DataFrame()
    forecast_df = pd.DataFrame()
    forecast_df['forecast_weather'] = weather_df['01_temp']
    for i in range(1,shift):
        forecast_df[f'forecast_weather_lag_{i+1}'] = forecast_df['forecast_weather'].shift(-i+1)
    
    weather_df = weather_df.shift(-shift)
# assuming df is your DataFrame
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