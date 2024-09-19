# Plot the demand over time
plt.figure(figsize=(14,7))
plt.plot(data['Demand'], label='Demand')
plt.title('Historical Demand Over Time')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.show()

# Decompose the time series to observe trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data['Demand'], model='additive')
decomposition.plot()
plt.show()
