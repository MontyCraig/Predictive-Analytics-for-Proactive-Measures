# Define SARIMA model order parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)  # Assuming monthly seasonality

# Fit the model
model = SARIMAX(train_data[target], order=order, seasonal_order=seasonal_order, exog=train_data[features])
model_fit = model.fit(disp=False)

# Summary of the model
print(model_fit.summary())
