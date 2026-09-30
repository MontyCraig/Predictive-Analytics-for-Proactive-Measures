# Basic Usage Guide

This guide provides a quick introduction to using the Predictive Analytics for Proactive Measures framework for common forecasting tasks.

## Overview

The framework provides a comprehensive solution for:

1. Collecting time series data from financial APIs

2. Performing exploratory data analysis

3. Training forecasting models

4. Generating predictions with confidence intervals

5. Identifying potential disruptions in the forecast

## Getting Started

After [installation](./installation.md), you can run the complete workflow using the main script:

```bash
<<<<<<< HEAD
python main.py --symbol MSFT
=======
predictive-analytics run --symbol MSFT
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will:

- Fetch historical data for Microsoft (MSFT) from Alpha Vantage

- Preprocess the data

- Run exploratory analysis

- Train a SARIMA model

- Generate a 30-day forecast

- Identify potential disruptions

- Save all outputs to the `output` directory

## Command-Line Options

Here are some common command-line options to customize the workflow:

```bash

# Analyze a different stock
<<<<<<< HEAD

python main.py --symbol AAPL

# Use a custom API key

python main.py --api-key YOUR_API_KEY

# Forecast a different time horizon

python main.py --forecast-steps 60

# Skip data collection (use previously downloaded data)

python main.py --skip-collection

# Show plots interactively instead of saving them

python main.py --show-plots

# Save outputs to a specific directory
=======
predictive-analytics run --symbol AAPL

# Use a custom API key
predictive-analytics run --api-key YOUR_API_KEY

# Forecast a different time horizon
predictive-analytics run --forecast-steps 60

# Skip data collection (use previously downloaded data)
predictive-analytics run --skip-collection

# Show plots interactively instead of saving them
predictive-analytics run --show-plots

# Save outputs to a specific directory
predictive-analytics run --output-dir /path/to/output
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

python main.py --output-dir /path/to/output

```text
For a complete list of options, run:

```bash
<<<<<<< HEAD
python main.py --help
=======
predictive-analytics run --help
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
## Step-by-Step Usage

### 1. Data Collection and Preprocessing

To only collect and preprocess data:

```bash
<<<<<<< HEAD
python -c "from utils.config import get_config; from data_collection_and_preprocessing import collect_and_preprocess_data; data = collect_and_preprocess_data(get_config(), symbol='AAPL')"
=======
python -c "from predictive_analytics.config.settings import get_config; from predictive_analytics.collection.preprocessing import collect_and_preprocess_data; data = collect_and_preprocess_data(get_config(), symbol='AAPL')"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will:

- Fetch historical data for Apple (AAPL)

- Preprocess the data

- Save it to `data/preprocessed/AAPL_preprocessed.csv`

### 2. Exploratory Data Analysis

To analyze the preprocessed data:

```bash
<<<<<<< HEAD
python -c "from utils.config import get_config; from exploratory_data_analysis import TimeSeriesExplorer; import pandas as pd; data = pd.read_csv('data/preprocessed/AAPL_preprocessed.csv', index_col=0, parse_dates=True); explorer = TimeSeriesExplorer(data); explorer.run_full_analysis()"
=======
python -c "from predictive_analytics.config.settings import get_config; from predictive_analytics.analysis.explorer import TimeSeriesExplorer; import pandas as pd; data = pd.read_csv('data/preprocessed/AAPL_preprocessed.csv', index_col=0, parse_dates=True); explorer = TimeSeriesExplorer(data); explorer.run_full_analysis()"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will create visualizations for:

- Time series plot

- Seasonal decomposition

- Autocorrelation and partial autocorrelation

- Rolling statistics

- Distribution analysis

- Correlation heatmap

### 3. Model Training

To train a forecasting model:

```bash
<<<<<<< HEAD
python -c "from utils.config import get_config; from model_selection_and_training import train_and_evaluate_model; metrics = train_and_evaluate_model(get_config(), file_path='data/preprocessed/AAPL_preprocessed.csv'); print(metrics)"
=======
python -c "from predictive_analytics.config.settings import get_config; from predictive_analytics.modeling.trainer import train_and_evaluate_model; metrics = train_and_evaluate_model(get_config(), file_path='data/preprocessed/AAPL_preprocessed.csv'); print(metrics)"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will:

- Load the preprocessed data

- Split it into training and test sets

- Train a SARIMA model

- Evaluate the model performance

- Save the trained model to `models/sarima_{timestamp}.pkl`

- Print evaluation metrics (RMSE, MAE, R², etc.)

### 4. Forecasting

To generate a forecast:

```bash
<<<<<<< HEAD
python -c "from utils.config import get_config; from forecasting_and_prediction import generate_forecast; forecast = generate_forecast(get_config(), data_path='data/preprocessed/AAPL_preprocessed.csv', steps=30); print(forecast.head())"
=======
python -c "from predictive_analytics.config.settings import get_config; from predictive_analytics.modeling.forecaster import generate_forecast; forecast = generate_forecast(get_config(), data_path='data/preprocessed/AAPL_preprocessed.csv', steps=30); print(forecast.head())"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will:

- Load the latest trained model (or train a new one if none exists)

- Generate a 30-day forecast with confidence intervals

- Save the forecast to `output/forecasts/data/forecast_30_steps_{timestamp}.csv`

- Save a plot of the forecast to `output/forecasts/plots/forecast_30_steps_{timestamp}.png`

- Print the first few rows of the forecast

### 5. Disruption Analysis

To identify potential disruptions:

```bash
<<<<<<< HEAD
python -c "from utils.config import get_config; from identifying_potential_disruptions import analyze_disruptions; disruption_df, report = analyze_disruptions(get_config(), forecast_path='output/forecasts/data/forecast_30_steps_{latest}.csv'); print(report)"
=======
python -c "from predictive_analytics.config.settings import get_config; from predictive_analytics.disruption.analyzer import analyze_disruptions; disruption_df, report = analyze_disruptions(get_config(), forecast_path='output/forecasts/data/forecast_30_steps_{latest}.csv'); print(report)"
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
This will:

- Load the latest forecast

- Identify trend, volatility, level, and uncertainty disruptions

- Save the disruption analysis to `output/disruptions/`

- Print a summary report

## Next Steps

After mastering the basic usage, you can:

- Learn about [advanced configuration options](./advanced_configuration.md)

- Explore the [API documentation](../api/index.md) for more customization

- Check out the guide on [adding new data sources](./adding_data_sources.md)

- Learn how to [add new forecasting models](./adding_models.md)
