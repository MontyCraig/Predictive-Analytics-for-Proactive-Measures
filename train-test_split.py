# Define features and target
features = ['Month', 'Weekday', 'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Rolling_Std_7']
target = 'Demand'

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
