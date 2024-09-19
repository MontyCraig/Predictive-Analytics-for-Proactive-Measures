# Load data
data = pd.read_csv('historical_demand.csv', parse_dates=['Date'], index_col='Date')

# Inspect data
print(data.head())

# Handle missing values
data = data.interpolate(method='time')
