# Using Facebook Prophet for Time Series Forecasting

This guide explains how to use Facebook Prophet for time series forecasting within the Predictive Analytics for Proactive Measures framework.

## Overview of Prophet

[Prophet](https://facebook.github.io/prophet/) is a forecasting procedure developed by Facebook for time series data. It's designed to handle time series with strong seasonal effects and several seasons of historical data. Prophet is robust to missing data, shifts in trends, and typically handles outliers well.

Key features:
- Automatic detection of seasonality
- Handles holidays and special events
- Robust to missing data and outliers
- Fast and scalable

## Prerequisites

Before using Prophet with this framework, ensure you have:

1. Installed Prophet and its dependencies:
   ```bash
   pip install prophet
   ```
   
   Note: Prophet may require additional system dependencies like `gcc`, `make`, and `Stan`. Refer to the [official installation guide](https://facebook.github.io/prophet/docs/installation.html) for detailed instructions.

2. Time series data in the appropriate format (the framework handles the conversion internally)

## Configuration

### Setting the Model Type

To use Prophet, set the model type to `prophet` in your `.env` file:

```
MODEL_TYPE=prophet
```

Alternatively, you can set it on the command line when running the model:

```bash
python model_selection_and_training.py --model prophet
```

### Prophet Configuration Options

Prophet offers several configuration options that you can set in your `.env` file:

```
# Prophet model configuration
PROPHET_YEARLY_SEASONALITY=True
PROPHET_WEEKLY_SEASONALITY=True
PROPHET_DAILY_SEASONALITY=False
PROPHET_SEASONALITY_MODE=additive  # Options: additive, multiplicative
PROPHET_CHANGEPOINT_PRIOR_SCALE=0.05
PROPHET_SEASONALITY_PRIOR_SCALE=10.0
```

#### Key Configuration Parameters

- **PROPHET_YEARLY_SEASONALITY**: Set to `True` to include yearly seasonality. Set to `False` to exclude it.
- **PROPHET_WEEKLY_SEASONALITY**: Set to `True` to include weekly seasonality.
- **PROPHET_DAILY_SEASONALITY**: Set to `True` for hourly data with daily patterns.
- **PROPHET_SEASONALITY_MODE**: 
  - `additive`: Use when the seasonal variations are roughly constant (default).
  - `multiplicative`: Use when the seasonal variations scale with the trend.
- **PROPHET_CHANGEPOINT_PRIOR_SCALE**: Controls flexibility of the trend. Higher values allow more flexibility (default: 0.05).
- **PROPHET_SEASONALITY_PRIOR_SCALE**: Controls flexibility of the seasonality. Higher values allow more flexibility (default: 10.0).

## Training a Prophet Model

### Using the Command Line

To train a Prophet model from the command line:

```bash
python model_selection_and_training.py --model prophet --symbol MSFT --exog volume,day_of_week
```

This command:
1. Sets the model type to Prophet
2. Uses Microsoft (MSFT) stock data
3. Includes 'volume' and 'day_of_week' as exogenous regressors

### Using the Python API

```python
from utils.config import get_config
from model_selection_and_training import train_and_evaluate_model

# Load configuration
config = get_config()

# Override model type
model_params = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "seasonality_mode": "additive",
}

# Train and evaluate Prophet model
metrics = train_and_evaluate_model(
    config,
    model_type="prophet",
    model_params=model_params,
    exog_columns=["volume", "day_of_week"],
    target_column="close"
)

print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

## Generating Forecasts with Prophet

### Using the Command Line

To generate a forecast using a trained Prophet model:

```bash
python forecasting_and_prediction.py --model-type prophet --steps 30
```

This command:
1. Loads the latest trained Prophet model
2. Generates a 30-day forecast

### Using the Python API

```python
from utils.config import get_config
from forecasting_and_prediction import generate_forecast

# Load configuration
config = get_config()

# Generate a 60-day forecast using a Prophet model
forecast_df = generate_forecast(
    config,
    model_type="prophet",
    steps=60,
    target_column="close",
    plot=True,
    save_forecast=True
)

print("Forecast Summary:")
print(forecast_df.head())

# Analyze the forecast
from forecasting_and_prediction import TimeSeriesForecaster

forecaster = TimeSeriesForecaster(config)
analysis = forecaster.analyze_forecast(forecast_df)

print("\nForecast Analysis:")
for metric, value in analysis.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")
```

## Complete Example Workflow

Here's a complete example showing how to collect data, train a Prophet model, and generate forecasts:

```python
from utils.config import get_config
from data_collection_and_preprocessing import collect_and_preprocess_data
from model_selection_and_training import ModelTrainer
from forecasting_and_prediction import TimeSeriesForecaster

# 1. Set up configuration
config = get_config()

# 2. Collect and preprocess data
data = collect_and_preprocess_data(config, symbol="AAPL", save=True)

# 3. Create a model trainer
trainer = ModelTrainer(config, data, target_column="close")

# 4. Split data into training and test sets
train_data, test_data = trainer.split_data(test_size=0.2)

# 5. Train a Prophet model
trainer.train_prophet_model(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    exog_columns=["volume", "day_of_week", "month"]
)

# 6. Evaluate the model
metrics = trainer.evaluate_model(exog_columns=["volume", "day_of_week", "month"])
print("\nModel Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 7. Save the model
from datetime import datetime
from pathlib import Path
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = Path(f'models/prophet_AAPL_{timestamp}.pkl')
trainer.save_model(model_path)

# 8. Create a forecaster
forecaster = TimeSeriesForecaster(config, model_path=model_path, data=data)

# 9. Generate a forecast
forecast_df = forecaster.forecast(steps=60)

# 10. Plot the forecast
forecaster.plot_forecast(forecast_df, historical_periods=90)

# 11. Analyze the forecast
analysis = forecaster.analyze_forecast(forecast_df)
print("\nForecast Analysis:")
for metric, value in analysis.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# 12. Detect potential disruptions
disruptions_df = forecaster.detect_anomalies(forecast_df, threshold=2.5)
print(f"\nDetected {disruptions_df['is_uncertain'].sum()} periods with high uncertainty")
```

## Advanced Usage

### Adding Custom Seasonality

Prophet allows you to add custom seasonality patterns. You can do this by extending the framework:

```python
# After creating a model trainer and before training
from model_selection_and_training import ModelTrainer
trainer = ModelTrainer(config, data)
trainer.split_data()

# Create a Prophet model with custom parameters
trainer.model = Prophet(
    yearly_seasonality=True, 
    weekly_seasonality=True
)

# Add custom quarterly seasonality
trainer.model.add_seasonality(
    name='quarterly',
    period=365.25/4,
    fourier_order=5
)

# Fit the model using our framework's data format
prophet_data = train_data.copy().reset_index()
prophet_data.rename(columns={prophet_data.columns[0]: 'ds', trainer.target_column: 'y'}, inplace=True)
trainer.model_fit = trainer.model.fit(prophet_data)

# Continue with evaluation, forecasting, etc.
```

### Including Holiday Effects

Prophet can account for holiday effects. Here's how to incorporate holidays:

```python
import pandas as pd
from datetime import datetime
from model_selection_and_training import ModelTrainer

# Define holidays
holidays = pd.DataFrame({
    'holiday': ['New Year', 'Independence Day', 'Thanksgiving', 'Christmas'],
    'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-11-23', '2023-12-25']),
    'lower_window': [0, 0, -1, -1],  # Days before the holiday
    'upper_window': [1, 1, 1, 1]     # Days after the holiday
})

# Create trainer and split data
trainer = ModelTrainer(config, data)
trainer.split_data()

# Create Prophet model with holidays
from prophet import Prophet
trainer.model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    holidays=holidays
)

# Continue with training and evaluation
prophet_data = train_data.copy().reset_index()
prophet_data.rename(columns={prophet_data.columns[0]: 'ds', trainer.target_column: 'y'}, inplace=True)
trainer.model_fit = trainer.model.fit(prophet_data)

# Evaluate and save as normal
```

## Troubleshooting

### Common Issues

1. **Installation Problems**: 
   - Prophet has several dependencies including Stan. If you encounter installation issues, refer to the [official installation guide](https://facebook.github.io/prophet/docs/installation.html).
   - For Linux, you may need to install: `apt-get install -y gcc make`

2. **Convergence Warnings**:
   - Prophet may show "Sampling did not converge" warnings. These are usually not critical but indicate the model might benefit from additional iterations.
   - Try adjusting the `PROPHET_SEASONALITY_PRIOR_SCALE` to a smaller value if you see convergence issues.

3. **Performance Issues**:
   - Prophet can be slower than other models, especially with large datasets.
   - Consider downsampling your data (e.g., from hourly to daily) if performance is an issue.

4. **Forecasting Far Future**:
   - Prophet's uncertainty bounds grow quickly when forecasting far into the future.
   - Be cautious when interpreting long-term forecasts.

## Resources

- [Official Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [Prophet Paper](https://peerj.com/preprints/3190/)
- [Prophet GitHub Repository](https://github.com/facebook/prophet)

## Next Steps

After mastering Prophet, you might want to explore:

- [Using Auto ARIMA](./using_auto_arima.md) for automated ARIMA model selection
- [Using Exponential Smoothing](./using_exp_smoothing.md) for simpler time series models
- [Advanced Configuration](./advanced_configuration.md) for fine-tuning model parameters 