# Calculate residuals
residuals = test_data[target] - forecast

# Plot residuals
plt.figure(figsize=(14,7))
plt.plot(residuals)
plt.title('Residuals of the Forecast')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Detect anomalies (e.g., residuals beyond 2 standard deviations)
std_dev = np.std(residuals)
anomalies = residuals[np.abs(residuals) > 2 * std_dev]

# Highlight anomalies
plt.figure(figsize=(14,7))
plt.plot(test_data[target], label='Actual Demand')
plt.scatter(anomalies.index, test_data.loc[anomalies.index][target], color='red', label='Anomalies')
plt.title('Anomalies in Demand')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.show()
