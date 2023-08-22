import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
import tqdm 
from weather_forecast import getWeatherForecastData

# Create prediction on a new data. Prediction dataset containes encoder and decoder data. With repeated data for all unavaliable
# future data
 
def tft_predict(X, X_train,  max_prediction_length, max_encoder_length, best_tft):
    
    #X.loc[X['timestamp'] >= '2022-07-31'].head(30)

    X = X.set_index('time_idx')


    # Select the last known data point
    # Filter the new data in X that extends beyond the last index of X_train

    last_index = X_train['time_idx'].max()
    new_data = X.loc[X['time_idx'] < last_index+max_prediction_length].copy()

    l = X['time_idx'].loc[(X['time_idx'] > last_index)].max()

    predicted_data = pd.DataFrame()
    current_timestamp = X_train['timestamp'].max()
    hour = 0

    for idx in tqdm(range(last_index,l)):
        encoder_data = X.loc[(X['time_idx'] >= idx - max_encoder_length) & (X['time_idx'] <= idx )].copy()
        # Select the last known data point
        last_data = encoder_data.loc[encoder_data['time_idx'] == idx]

        decoder_data = pd.concat(
        [last_data.assign(
            time_idx=last_data['time_idx'] + i) for i in range(1, max_prediction_length + 1)], ignore_index=True)

        # List of columns to be replaced
        columns_to_replace = ["timestamp", "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday", "weekday_binary","holiday","holiday_lag", "holiday_lead", "load_lag_24", "load_lag_48", "load_lag_168", "fct_temp"]

        # Define the condition for the rows you want to modify in X
        condition_X = (X['time_idx'] > idx) & (X['time_idx'] <= idx + max_prediction_length)

        # Select the rows from X that match the condition and set 'time_idx' as the index
        X_temp = X.loc[condition_X].set_index('time_idx')

        # Create a copy of decoder_data with 'time_idx' as the index
        decoder_data_temp = decoder_data.set_index('time_idx')

        # Replace the specified columns in decoder_data_temp with the corresponding columns in X_temp
        decoder_data_temp.loc[X_temp.index, columns_to_replace] = X_temp[columns_to_replace]
        #print(decoder_data_temp)
        fromDate = decoder_data['timestamp'].iloc[0].strftime('%Y-%m-%d')
        toDate = decoder_data['timestamp'].iloc[-1].strftime('%Y-%m-%d')
        
        forecast_df = getWeatherForecastData(True, fromDate, toDate)
        decoder_data['fct_temp'] = forecast_df['fct_temp']
    
        # Reset the index in decoder_data_temp to merge the changes back into decoder_data
        decoder_data = decoder_data_temp.reset_index()

    
        # Combine your encoder data and decoder data
        new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

        #new_prediction_data['timestamp'] = pd.to_datetime(new_prediction_data['timestamp'], utc=True)

        new_prediction_data['time_idx'] = new_prediction_data['time_idx'].astype(int)



        # Use your model to make predictions on the new data
        predicted_values = best_tft.predict(new_prediction_data, return_y=True, return_x=True)

        actual_values = predicted_values.y[0].numpy().flatten()
        predicted_values = predicted_values.output.numpy().flatten()
    

        # Transform data in one dataframe
        predicted_data_temp = pd.DataFrame({'prediction': predicted_values}, index = decoder_data['time_idx'])
        predicted_data_temp['timestamp'] = current_timestamp + pd.DateOffset(hours=hour)
    
        predicted_data_temp['forecasted_timestamp'] = decoder_data_temp['timestamp']

        predicted_data_temp = pd.merge(predicted_data_temp, X[['time_idx', 'Actual Load']], how='left', on='time_idx')

        # Put all predicted values together 
        predicted_data = pd.concat([predicted_data, predicted_data_temp])
        predicted_data.to_csv('check.csv' )
        hour = hour + 1

    predicted_data['time_difference'] = (pd.to_datetime(predicted_data["forecasted_timestamp"]) - pd.to_datetime(predicted_data['timestamp'])).dt.total_seconds() / 3600
    return predicted_data

def printMetricsHourly(predicted_data, max_prediction_length):
    for idx in range(1,max_prediction_length):
        selected_rows = predicted_data.loc[predicted_data['time_difference'] == idx]
        print('MAPE '+ str(idx) + ' hour : ' + str(mean_absolute_percentage_error(selected_rows['Actual Load'], selected_rows['prediction'])))
        print('MSE '+ str(idx) + ' hour : ' + str(mean_squared_error(selected_rows['Actual Load'], selected_rows['prediction'])))
        print('MAE '+ str(idx) + ' hour : ' + str(mean_absolute_error(selected_rows['Actual Load'], selected_rows['prediction'])))
        print('R^2 score '+ str(idx) + ' hour : '+ str(r2_score(selected_rows['Actual Load'], selected_rows['prediction'])))
        print('------------------------------------------------------------------------------------------------------------')
    
