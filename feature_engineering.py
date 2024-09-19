# Create time-based features
data['Month'] = data.index.month
data['Weekday'] = data.index.weekday
data['Year'] = data.index.year

# Lag features
data['Lag_1'] = data['Demand'].shift(1)
data['Lag_7'] = data['Demand'].shift(7)
data['Lag_30'] = data['Demand'].shift(30)

# Rolling window statistics
data['Rolling_Mean_7'] = data['Demand'].rolling(window=7).mean()
data['Rolling_Std_7'] = data['Demand'].rolling(window=7).std()

# Drop rows with NaN values resulting from shifting
data = data.dropna()
