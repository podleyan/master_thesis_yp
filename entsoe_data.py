from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np

# Get electricity load data for specific country in date range fromDate toDate
# API key retrieved from https://www.entsoe.eu/

def getEntsoeData(location, fromDate, toDate, type):
    start = pd.Timestamp(str(fromDate), tz ='UTC')
    end = pd.Timestamp(str(toDate), tz ='UTC')
  
    client = EntsoePandasClient(api_key='')
    data = pd.date_range(start=start, end=end, freq='H', tz='UTC')
    entsoe_all = pd.DataFrame(index=data)
    country_code = str(location)
    if str(type) == 'history':
        load = client.query_load(country_code, start=start, end=end)
    elif str(type) == 'forecast':
        load = client.query_load_forecast(country_code, start=start, end=end)
    else:
        print('Wrong type selected, please select type \in (history, forecast)')
        
    load = pd.merge(load, entsoe_all, how='outer', left_index=True, right_index=True)
        
    load = load.resample('H').sum()
    load['country'] = location
    load['Actual Load'] = load['Actual Load'].replace(0, np.nan)
    load['Actual Load'] = load['Actual Load'].interpolate()
    return load

# EXAMPLE:
#data = getEntsoeData('CZ', 20171201, 20181201, 'history')
#print(data)

