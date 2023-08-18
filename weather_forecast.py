import requests
import pandas as pd

# Get weather forecast for specific location in data range for prediction on a new data, load from csv or Open Meteo API. 
# Open Meteo have historical forecast data only for last three month.
 
def getWeatherForecastData(load_from_csv, fromDate, toDate):
    if load_from_csv:
        df = pd.read_csv('weather_forecast.csv', sep=';')
        return df
    else:
        # Open-Meteo API endpoint
        api_url = "https://api.open-meteo.com/v1/forecast"
    
        # For data from Prague
        latitude = 50.088
        longitude = 14.4208

        # Parameters for the API request
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m',
            'start_date': fromDate,
            'end_date': toDate
        }
    
        # Send a GET request to the Open-Meteo API
        response = requests.get(api_url, params=params)
    
        # Check for successful response
        if response.status_code == 200:
            # Parse the JSON data from the response
            data = response.json()
            hourly_data = data['hourly']
            df = pd.DataFrame.from_dict(hourly_data)
            df = df.rename(columns={'time': 'timestamp', 'temperature_2m': 'fct_temp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Return the parsed data
            return df
    
        else:
            # Print an error message if the request was unsuccessful
            print("Failed to retrieve data. Status code:", response.status_code)
            return None
    
data = getWeatherForecastData(1, '2023-06-01', '2023-06-02')
print(data)