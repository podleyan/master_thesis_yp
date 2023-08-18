import pandas as pd 
#######################################################################################################################
# Get graph for ensemble model 

ensemble1 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe1.csv')
ensemble2 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe2.csv')
ensemble3 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe3.csv')
ensemble4 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe4.csv')
ensemble5 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe5.csv')
ensemble6 = pd.read_csv('/Users/yanapodlesna/main/skool/master/ensembe6.csv')
ensemble1['forecast_time'] = pd.to_datetime(ensemble1['timestamp']) - pd.to_timedelta(1, unit='h')
ensemble2['forecast_time'] = pd.to_datetime(ensemble2['timestamp']) - pd.to_timedelta(2, unit='h')
ensemble3['forecast_time'] = pd.to_datetime(ensemble3['timestamp']) - pd.to_timedelta(3, unit='h')
ensemble4['forecast_time'] = pd.to_datetime(ensemble4['timestamp']) - pd.to_timedelta(4, unit='h')
ensemble5['forecast_time'] = pd.to_datetime(ensemble5['timestamp']) - pd.to_timedelta(5, unit='h')
ensemble6['forecast_time'] = pd.to_datetime(ensemble6['timestamp']) - pd.to_timedelta(6, unit='h')

all = ensemble1.append(ensemble2)
all = all.append(ensemble3)
all = all.append(ensemble4)
all = all.append(ensemble5)
all = all.append(ensemble6)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Same as before
figure_date = '2023-06-01 00:00:00+00:00'
subset = ((pd.to_datetime(all['forecast_time']) == figure_date))
figure_df = all.loc[subset].copy()

# Convert 'timestamp' to datetime
figure_df['timestamp'] = pd.to_datetime(figure_df['timestamp'])

# Create a new figure and set the size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the historical load
ax.plot(figure_df['timestamp'], figure_df['Actual Load'], label="Historical load")

# Plot the prediction
ax.plot(figure_df['timestamp'], figure_df['Prediction'], label="Prediction")

# Add title and labels
ax.set_xlabel("Time")
ax.set_ylabel("Load")

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))
fig.autofmt_xdate()

# Add a legend
ax.legend()

# Show the plot
plt.show()
