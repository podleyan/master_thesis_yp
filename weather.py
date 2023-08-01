## Import Meteostat library and dependencies
from meteostat import Point, Daily, Hourly
import pandas as pd

def getWeatherData(country, fromDate, toDate, type):
    # Set time period
    start = pd.to_datetime(str(fromDate), format='%Y%m%d')
    end = pd.to_datetime(str(toDate), format='%Y%m%d')
    
    # Creating dictionary of locations (biggest cities in country)
    if 'CZ' in country:
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
    if 'SK' in country:
        locations = {
        # Slovakia:
        '01' : Point(48.1486, 17.1077, 134), # Bratislava
        '02' : Point(48.7200, 21.2571, 206), # Košice
        '03' : Point(48.9988, 21.2401, 250), # Prešov
        '04' : Point(49.2225, 18.7403, 342), # Žilina
        '05' : Point(48.3091, 18.0873, 135), # Nitra
        '06' : Point(48.7333, 19.1500, 362), # Banská Bystrica
        '07' : Point(48.3774, 17.5883, 146), # Trnava
        '08' : Point(49.0678, 18.9322, 395), # Martin
        '09' : Point(48.8945, 18.0447, 225), # Trenčín
        '10' : Point(49.0614, 20.2975, 672), # Poprad
        }
        
    if 'AT' in country:
        locations = {
            # Austria:
            'AT01' : Point(48.208174,16.373818,165), # Vienna
            'AT02' : Point(47.076668,15.421371,353), # Graz
            'AT03' : Point(48.306940,14.285832,261), # Linz
            'AT04' : Point(47.809493,13.055012,431), # Salzburg
            'AT05' : Point(47.269212,11.404102,431), # Innsbruck
            'AT06' : Point(47.269212,11.404102,576), # Klagenfurt am Worthersee
            'AT07' : Point(46.608563,13.850625,497), # Villach (since 2018)
            'AT08' : Point(48.165421,14.036643,313), # Wels
            'AT09' : Point(48.203537,15.638171,269), # Sankt Polten
            'AT10' : Point(47.412415,9.743791,440), # Dornbirn
        }
    if 'DE' in country:
        locations = {
            '01' : Point(52.520008,13.404954,34), #Berlin
            '02' : Point(53.551086,9.993682,6), #Hamburg
            '03' : Point(48.137154,11.576124,522), #Munich
            '04' : Point(49.500983,11.051859,312), #Nurnberg
            '05' : Point(48.783333,9.183333,250), #Stuttgart
            '06' : Point(53.073635,8.806422,15), #Bremen
            '07' : Point(51.050407,13.737262,117), #Dresden
            '08' : Point(53.396520,9.662903,50), #Hannover
            '09' : Point(51.450832,7.013056,28), #Essen
            '10' : Point(50.110924,8.682127,106), #Frankfurt am Main
        }
    if 'HU' in country:
        locations = {
    # Hungary:
            '01' : Point(47.4979, 19.0402, 96), # Budapest
            '02' : Point(47.5315, 21.6273, 120), # Debrecen
            '03' : Point(46.2530, 20.1414, 75), # Szeged
            '04' : Point(46.0767, 18.2281, 173), # Pécs
            '05' : Point(47.6875, 17.6504, 112), # Győr
            '06' : Point(48.1035, 20.7784, 118), # Miskolc
            '07' : Point(47.1865, 18.4221, 125), # Székesfehérvár
            '08' : Point(46.8964, 19.6897, 120), # Kecskemét
            '09' : Point(47.6817, 16.5845, 232), # Sopron
            '10' : Point(47.2307, 16.6218, 209), # Szombathely
        }
    # Creating dataset for location
    df = pd.DataFrame()
    for key in locations:
        if type == 'daily':
            data = Daily(locations[key], start, end)
        if type == 'hourly':
            data = Hourly(locations[key], start, end)
        data = data.fetch()
        # data.insert(loc=0, column='location', value=key)
        data = data.add_prefix('{}_'.format(key))
        df = pd.concat([df,data],axis=1)
    
    df.index = pd.to_datetime(df.index.values, utc=True)
    data = pd.date_range(start=start, end=end, freq='H', tz='UTC')
    weather_all = pd.DataFrame(index=data)
    df_merged = pd.merge(df, weather_all, how='outer', left_index=True, right_index=True)
    df = df_merged.interpolate()
    
    return df