import pandas as pd
import holidays

def convert_weekday(day):
    if day >= 1 and day <= 5:
        return 1  # Workday
    else:
        return 0  # Weekend


def getCalendarData(location, fromDate, toDate):
    start = pd.to_datetime(str(fromDate), format='%Y%m%d')
    end = pd.to_datetime(str(toDate), format='%Y%m%d')

    data = pd.date_range(start=start, end=end, freq='D')
    calendar_df = pd.DataFrame(columns=['date', 'country', 'region', 'weekday', 'month', 'holiday'])
    df = calendar_df 
    #calendar_df.columns  = ['date', 'country', 'region', 'weekday', 'month', 'holiday']
    

    if 'CZ' in location:
        cz_holidays = holidays.Czechia()
        df['date'] = data
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['country'] = 'CZ'
        df['holiday'] = [1 if str(val).split()[0] in cz_holidays else 0 for val in df['date']]
        df['weekday_binary'] = df['weekday'].apply(convert_weekday)
        df['holiday_lag'] = df['holiday'].shift(1)
        df['holiday_lead'] = df['holiday'].shift(-1)
        df.at[0, 'holiday_lag'] = 1
        calendar_df = pd.concat([calendar_df, df], ignore_index=True)

    if 'SK' in location:
        sk_holidays = holidays.Slovakia()
        df['date'] = data
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['country'] = 'SK'
        df['holiday'] = [1 if str(val).split()[0] in sk_holidays else 0 for val in df['date']]
        df['weekday_binary'] = df['weekday'].apply(convert_weekday)
        df['holiday_lag'] = df['holiday'].shift(1)
        df['holiday_lead'] = df['holiday'].shift(-1)
        df.at[0, 'holiday_lag'] = 1
        calendar_df = pd.concat([calendar_df, df], ignore_index=True)

    if 'HU' in location:
        hu_holidays = holidays.Hungary()
        df['date'] = data
        df['weekday'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['country'] = 'HU'
        df['holiday'] = [1 if str(val).split()[0] in hu_holidays else 0 for val in df['date']]
        df['weekday_binary'] = df['weekday'].apply(convert_weekday)
        df['holiday_lag'] = df['holiday'].shift(1)
        df['holiday_lead'] = df['holiday'].shift(-1)
        df.at[0, 'holiday_lag'] = 1
        calendar_df = pd.concat([calendar_df, df], ignore_index=True)

    if 'DE' in location:
        provinces = ['BB', 'BE', 'BW', 'BY', 'HB', 'HE', 'HH', 'MV', 'NI', 'NW', 'SH', 'RP', 'SL', 'SN', 'ST', 'TH']

        for prov in provinces:
            prov_holiday = holidays.Germany(prov=prov)
            df['date'] = data
            df['weekday'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['country'] = 'DE'
            df['region'] = prov
            df['holiday'] = [1 if str(val).split()[0] in prov_holiday else 0 for val in df['date']]
            calendar_df = pd.concat([calendar_df, df], ignore_index=True)

    if 'AT' in location:
        for prov in range(1, 9):
            prov_holiday = holidays.Austria(prov=prov)
            df['date'] = data
            df['weekday'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['country'] = 'AT'
            df['region'] = prov
            df['holiday'] = [1 if str(val).split()[0] in prov_holiday else 0 for val in df['date']]
            calendar_df = pd.concat([calendar_df, df], ignore_index=True)


    return calendar_df
