## Import Meteostat library and dependencies
import logging

# Set up logging
logging.basicConfig(filename='/Users/yanapodlesna/main/skool/master/weather_forecast.log', level=logging.DEBUG)

logging.debug('Script started')

try:
    from meteostat import Point, Daily, Hourly
    from datetime import datetime, timedelta
    import pandas as pd

    logging.debug('Libraries imported')

    # Get current date and time (rounded to the nearest hour)
    now = datetime.now()
    fromDate = now.replace(minute=0, second=0, microsecond=0)

    # Calculate toDate as fromDate + 36h
    toDate = fromDate + timedelta(hours=36)

    # Set time period
    start = pd.to_datetime(fromDate+ timedelta(hours=1))
    end = pd.to_datetime(toDate)

    logging.debug(f'Start time: {start}, End time: {end}')

    # Rest of your code...

except Exception as e:
    logging.error(f'Error: {e}')

# Rest of your code...

locations = {
    # Czech Republic:
    '01' : Point(50.075539,14.437800,239), # Praha
    '02' : Point(49.195060,16.606837,237), # Brno
    '03' : Point(49.820923,18.262524,260), # Ostrava
    '04' : Point(49.738431,13.373637,310), # Plzen (since 2018)
    '05' : Point(50.766280,15.054339,374), # Liberec
    '06' : Point(49.593778,17.250879,219), # Olomouc
    '07' : Point(48.975658,14.480255,381), # Ceske Budejovice (since 2018)
    '08' : Point(50.661116,14.053146,218), # Usti nad Labem
    '09' : Point(50.210361,15.825211,235), # Hradec Kralove
    '10' : Point(50.034309,15.781199,237), # Pardubice
}

df = pd.DataFrame()

for key in locations:
    data = Hourly(locations[key], start, end)
    data = data.fetch()
    data = data.add_prefix('{}_'.format(key))
    df = pd.concat([df,data],axis=1)

df['fromDate'] = fromDate
df.index = pd.to_datetime(df.index.values, utc=True)

logging.debug('Data fetched and processed')

# ...

try:
    df.to_csv('/Users/yanapodlesna/main/skool/master/forecast.csv', mode='a', header=False)
    logging.debug('Data saved to forecast.csv')
except Exception as e:
    logging.error(f'Error while saving data to CSV: {e}')

logging.debug('Script ended')



