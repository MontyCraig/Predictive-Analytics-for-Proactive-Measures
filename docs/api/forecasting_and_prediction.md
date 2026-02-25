# Forecasting and Prediction Module

## Overview

The Forecasting and Prediction module provides a comprehensive framework for generating time series forecasts using trained models. It includes functionality for generating forecasts with confidence intervals, visualizing results, and analyzing forecast characteristics.

## Architecture

The module follows a clean, object-oriented architecture with these primary components:

1. **Forecaster Class**: Core class that handles forecast generation
2. **Visualization Tools**: Methods for plotting forecasts with historical data
3. **Forecast Analysis**: Functions for analyzing forecast properties
4. **Integration Layer**: Functions to simplify the overall forecasting workflow

## API Reference

### `TimeSeriesForecaster` (Class)

Forecaster for time series data using trained models.

#### `__init__(config: Config, model_path: Optional[Union[str, Path]] = None, data: Optional[pd.DataFrame] = None, target_column: str = "close") -> None`

Initialize the forecaster.

**Parameters:**
- `config` (Config): Application configuration
- `model_path` (Union[str, Path], optional): Path to trained model file
- `data` (pd.DataFrame, optional): Time series data
- `target_column` (str, optional): Target column to predict (default: "close")

#### `load_model(file_path: Union[str, Path]) -> None`

Load a trained model from a file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the model file

**Raises:**
- `FileNotFoundError`: If the model file doesn't exist
- `Exception`: For other errors during loading

#### `load_data(file_path: Union[str, Path]) -> None`

Load data from a CSV file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the CSV file

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `Exception`: For other errors during loading

#### `forecast(steps: int = 30, exog_features: Optional[pd.DataFrame] = None, confidence_interval: float = 0.95) -> pd.DataFrame`

Generate forecasts for future time periods.

**Parameters:**
- `steps` (int, optional): Number of steps to forecast (default: 30)
- `exog_features` (pd.DataFrame, optional): Exogenous features for the forecast period
- `confidence_interval` (float, optional): Confidence interval for prediction intervals (default: 0.95)

**Returns:**
- `pd.DataFrame`: DataFrame with forecasts and prediction intervals

**Raises:**
- `ValueError`: If model is not loaded
- `Exception`: For other errors during forecasting

#### `plot_forecast(forecast_df: pd.DataFrame, historical_periods: int = 60, save_path: Optional[Union[str, Path]] = None) -> None`

Plot forecasts with historical data and prediction intervals.

**Parameters:**
- `forecast_df` (pd.DataFrame): DataFrame with forecasts from the forecast method
- `historical_periods` (int, optional): Number of historical periods to show (default: 60)
- `save_path` (Union[str, Path], optional): Path to save the plot

#### `detect_anomalies(forecast_df: pd.DataFrame, actual_values: Optional[pd.Series] = None, threshold: float = 2.0) -> pd.DataFrame`

Detect anomalies in the forecast or actual values.

**Parameters:**
- `forecast_df` (pd.DataFrame): DataFrame with forecasts
- `actual_values` (pd.Series, optional): Actual values for the forecast period
- `threshold` (float, optional): Z-score threshold for anomaly detection (default: 2.0)

**Returns:**
- `pd.DataFrame`: DataFrame with anomalies flagged

#### `analyze_forecast(forecast_df: pd.DataFrame) -> Dict[str, float]`

Analyze the forecast for key indicators.

**Parameters:**
- `forecast_df` (pd.DataFrame): DataFrame with forecasts

**Returns:**
- `Dict[str, float]`: Dictionary with forecast analysis metrics

### `generate_forecast(config: Config, model_path: Optional[Union[str, Path]] = None, data_path: Optional[Union[str, Path]] = None, steps: int = 30, target_column: str = "close", plot: bool = True, save_forecast: bool = True, save_plot: bool = True) -> pd.DataFrame`

Generate and analyze forecasts.

**Parameters:**
- `config` (Config): Application configuration
- `model_path` (Union[str, Path], optional): Path to trained model file
- `data_path` (Union[str, Path], optional): Path to data file
- `steps` (int, optional): Number of steps to forecast (default: 30)
- `target_column` (str, optional): Target column to predict (default: "close")
- `plot` (bool, optional): Whether to plot the forecast (default: True)
- `save_forecast` (bool, optional): Whether to save the forecast to a CSV file (default: True)
- `save_plot` (bool, optional): Whether to save the plot (default: True)

**Returns:**
- `pd.DataFrame`: DataFrame with forecasts and analysis

## Usage Examples

### Basic Usage

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.forecaster import generate_forecast

# Load configuration
config = get_config()

# Generate forecast
forecast_df = generate_forecast(
    config,
    steps=30,
    target_column="close"
)
```

### Customized Forecasting

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster
from pathlib import Path

# Load configuration
config = get_config()

# Create forecaster with specific model
forecaster = TimeSeriesForecaster(
    config,
    model_path="models/sarima_20240101_120000.pkl",
    target_column="close"
)

# Load historical data
forecaster.load_data("data/preprocessed/AAPL_preprocessed.csv")

# Generate forecast with custom parameters
forecast_df = forecaster.forecast(
    steps=60,
    confidence_interval=0.90
)

# Plot forecast
forecaster.plot_forecast(
    forecast_df,
    historical_periods=90,
    save_path="output/forecasts/plots/aapl_60day_forecast.png"
)

# Analyze forecast
metrics = forecaster.analyze_forecast(forecast_df)
print(f"Average daily change: {metrics['avg_daily_change']:.4f}")
print(f"Growth rate: {metrics['growth_rate_pct']:.2f}%")
print(f"Volatility: {metrics['volatility']:.4f}")
```

### Anomaly Detection

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster
import pandas as pd

# Load configuration
config = get_config()

# Create forecaster
forecaster = TimeSeriesForecaster(
    config,
    model_path="models/sarima_20240101_120000.pkl"
)

# Load historical data
forecaster.load_data("data/preprocessed/MSFT_preprocessed.csv")

# Generate forecast
forecast_df = forecaster.forecast(steps=30)

# Detect anomalies in the forecast
anomalies_df = forecaster.detect_anomalies(
    forecast_df,
    threshold=2.5
)

# Filter to show only uncertain periods
uncertain_periods = anomalies_df[anomalies_df["is_uncertain"]]
print(f"Detected {len(uncertain_periods)} periods with high uncertainty")
```

## Implementation Details

### Forecast Generation

The module uses trained time series models to generate forecasts with the following components:

1. **Point Forecasts**: The expected future values of the time series
2. **Confidence Intervals**: Upper and lower bounds representing the uncertainty of the forecast

### Visualization Features

The forecasting module includes comprehensive visualization capabilities:

1. **Historical and Forecast Plot**: Shows historical data alongside forecasts
2. **Confidence Intervals**: Visualizes the uncertainty in the forecast
3. **Anomaly Highlighting**: Can mark potential anomalies or periods of high uncertainty

### Forecast Analysis

The module provides several analytical capabilities:

1. **Trend Analysis**: Calculation of average daily change and growth rate
2. **Volatility Assessment**: Measurement of forecast volatility
3. **Uncertainty Quantification**: Analysis of the width of prediction intervals
4. **Extrema Identification**: Detection of minimum and maximum forecasted values

## Integration Points

The module integrates with:
- Configuration management system for model parameters
- Model selection and training module for loading trained models
- Data preprocessing module for historical data
- File system for forecast persistence
- Logging system for operational insights

## Areas for Enhancement

1. **Probabilistic Forecasting**: Extend to support fully probabilistic forecasts beyond confidence intervals
2. **Ensemble Forecasting**: Implement methods to combine forecasts from multiple models
3. **Scenario Analysis**: Add support for scenario-based forecasting with different assumptions
4. **Dynamic Updating**: Implement functionality to update forecasts as new data becomes available
5. **Long-Range Forecasting**: Improve accuracy for longer forecast horizons
6. **Hierarchical Forecasting**: Add support for forecasting hierarchical time series data 