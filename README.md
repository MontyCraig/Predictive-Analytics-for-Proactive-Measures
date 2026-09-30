# Predictive Analytics for Proactive Measures

Enterprise-grade time series forecasting and disruption prediction framework.

## Overview

A modular Python package for collecting market data, training forecasting models, and identifying potential disruptions before they impact business operations. Supports SARIMA, Prophet, Auto ARIMA, and Exponential Smoothing models with fully typed Pydantic v2 configuration.

<<<<<<< HEAD
1. Forecast future values with statistical confidence intervals

2. Identify potential disruptions and anomalies in advance

3. Provide actionable insights for proactive business measures

The implementation follows the project's Python coding standards (see `docs/coding_standards/`), with strong typing, modular architecture, comprehensive error handling, and thorough documentation.

## Features

- **Data Collection & Preprocessing**: Integrates with Alpha Vantage API (and extendable to other sources) to fetch real-time market data with robust error handling and data validation

- **Exploratory Data Analysis**: Comprehensive time series analysis tools including seasonal decomposition, stationarity testing, and correlation analysis

- **Model Training & Evaluation**: SARIMA model implementation with configurable parameters and extensive evaluation metrics

- **Forecasting & Prediction**: Generate forecasts with confidence intervals and uncertainty quantification

- **Disruption Identification**: Advanced algorithms to detect trend, volatility, level, and uncertainty disruptions

## Project Structure

```text
predictive-analytics/
├── data/                     # Data storage directory

├── models/                   # Saved model files

├── output/                   # Generated outputs

│   ├── raw/                  # Raw data

│   ├── preprocessed/         # Preprocessed data

│   ├── plots/                # Generated plots

│   ├── models/               # Trained models

│   ├── forecasts/            # Forecast outputs

│   └── disruptions/          # Disruption analysis

├── utils/                    # Utility modules

│   ├── __init__.py
│   ├── config.py             # Configuration management
=======
### Key Capabilities

- **Data Collection** -- Alpha Vantage API integration with rate-limit handling and data validation
- **Exploratory Data Analysis** -- Seasonal decomposition, stationarity testing, ACF/PACF, distribution analysis
- **Model Training** -- Four model types with configurable hyperparameters and evaluation metrics
- **Forecasting** -- Predictions with confidence intervals and anomaly detection
- **Disruption Detection** -- Trend, volatility, level-shift, and uncertainty disruption identification

## Package Structure

```
src/predictive_analytics/
    __init__.py              # Public API: AppConfig, ModelTrainer, etc.
    __main__.py              # python -m predictive_analytics
    py.typed                 # PEP 561 type-checking marker
    cli.py                   # Typer CLI (predictive-analytics command)
    exceptions.py            # Custom exception hierarchy
    types.py                 # Shared TypeAlias definitions
    collection/
        client.py            # AlphaVantageClient
        preprocessing.py     # Data cleaning and feature engineering
    analysis/
        explorer.py          # TimeSeriesExplorer (EDA)
    modeling/
        trainer.py           # ModelTrainer (4 model types)
        forecaster.py        # TimeSeriesForecaster
    disruption/
        analyzer.py          # DisruptionAnalyzer (4 disruption types)
    config/
        settings.py          # Pydantic v2 configuration models
        logging.py           # Structured logging setup
    utils/
        helpers.py           # String, datetime, file, and data utilities
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

│   └── logging_config.py     # Logging setup

├── data_collection_and_preprocessing.py   # Data collection module

├── exploratory_data_analysis.py           # EDA module

├── model_selection_and_training.py        # Model training module

├── forecasting_and_prediction.py          # Forecasting module

├── identifying_potential_disruptions.py   # Disruption detection module

├── main.py                                # Main entry point

├── requirements.txt                       # Dependencies

└── README.md                              # Documentation

```text
## Installation

<<<<<<< HEAD
1. Clone the repository:

   ```text
   git clone <https://github.com/your-username/predictive-analytics-for-proactive-measures.git>
   cd Predictive-Analytics-for-Proactive-Measures
   ```text

2. Create and activate a conda environment:

   ```text
   conda create -n predictive_analytics python=3.9
   conda activate predictive_analytics
   ```text

3. Install dependencies:

   ```text
   pip install -r requirements.txt
   ```text

4. Get an API key from [Alpha Vantage](<https://www.alphavantage.co/support/#api-key)> for data collection (a free key is available)
=======
```bash
# Clone the repository
git clone https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures.git
cd Predictive-Analytics-for-Proactive-Measures

# Create conda environment
conda create -n predictive-analytics python=3.12
conda activate predictive-analytics

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

### CLI Usage

```bash
<<<<<<< HEAD
python main.py --symbol MSFT --api-key YOUR_API_KEY

```text
This will:

1. Collect and preprocess stock data for Microsoft (MSFT)

2. Perform exploratory data analysis with visualizations

3. Train a SARIMA model and evaluate its performance

4. Generate forecasts for the next 30 days

5. Analyze the forecast for potential disruptions

### Command Line Options

```text
usage: main.py [-h] [--symbol SYMBOL] [--api-key API_KEY] [--env-file ENV_FILE]
               [--output-dir OUTPUT_DIR] [--skip-collection] [--skip-eda]
               [--skip-training] [--skip-forecasting] [--skip-disruptions]
               [--show-plots] [--forecast-steps FORECAST_STEPS]

Predictive Analytics for Proactive Measures

optional arguments:
  -h, --help            show this help message and exit
  --symbol SYMBOL       Stock symbol to analyze (default: MSFT)
  --api-key API_KEY     Alpha Vantage API key
  --env-file ENV_FILE   Path to .env file with API keys
  --output-dir OUTPUT_DIR
                        Directory to save outputs
  --skip-collection     Skip data collection
  --skip-eda            Skip exploratory data analysis
  --skip-training       Skip model training
  --skip-forecasting    Skip forecasting
  --skip-disruptions    Skip disruption analysis
  --show-plots          Show plots instead of saving
  --forecast-steps FORECAST_STEPS
                        Number of steps to forecast (default: 30)

```text
### Using Individual Modules

You can also use each module independently for more granular control:

```python

# Example: Just collect and preprocess data

from utils.config import get_config
from data_collection_and_preprocessing import collect_and_preprocess_data
=======
# Run the full pipeline for a stock symbol
predictive-analytics run MSFT --api-key YOUR_KEY

# Skip specific stages
predictive-analytics run AAPL --skip-collection --skip-eda

# Custom forecast horizon
predictive-analytics run GOOG --forecast-steps 60 --output-dir ./results
```

### Python API

```python
from predictive_analytics import AppConfig, AlphaVantageClient, ModelTrainer
from predictive_analytics.config.settings import get_config
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

# Load configuration from environment / .env file
config = get_config()
<<<<<<< HEAD
data = collect_and_preprocess_data(config, symbol="AAPL")
=======

# Collect and preprocess data
from predictive_analytics.collection.preprocessing import collect_and_preprocess_data
data = collect_and_preprocess_data(config, symbol="MSFT")

# Train a model
trainer = ModelTrainer(config, data=data, target_column="close")
trainer.train_sarima()
metrics = trainer.evaluate_model()

# Generate forecast
from predictive_analytics.modeling.forecaster import generate_forecast
forecast_df = generate_forecast(config, steps=30, target_column="close")
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

```text
## Configuration

Configuration is managed through Pydantic v2 models with environment variable support:

<<<<<<< HEAD
1. Environment variables

2. A `.env` file in the project root

3. Command-line arguments
=======
```bash
# Required
export ALPHA_VANTAGE_API_KEY=your_key_here
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

# Optional (with defaults)
export MODEL_TYPE=sarima          # sarima | prophet | auto_arima | exp_smoothing
export TRAIN_SIZE=0.8
export TEST_SIZE=0.2
export LOG_LEVEL=INFO
```

<<<<<<< HEAD
- `ALPHA_VANTAGE_API_KEY`: Your API key for data collection

- `MODEL_TYPE`: Forecasting model type (default: "sarima")

- `TRAIN_SIZE`: Proportion of data for training (default: 0.8)

- `TEST_SIZE`: Proportion of data for testing (default: 0.2)

- `LOG_LEVEL`: Logging verbosity (default: "INFO")
=======
Or use a `.env` file (see `.env-example` for all available options).
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (requires 100% coverage)
pytest

<<<<<<< HEAD
1. Create a new client class similar to `AlphaVantageClient`

2. Implement the required methods for data fetching and preprocessing

3. Update the configuration system to support the new data source
=======
# Code quality
black src/ tests/
isort src/ tests/
mypy --strict src/
flake8 src/ tests/
bandit -r src/ -c pyproject.toml
```
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Documentation

<<<<<<< HEAD
1. Extend the `ModelTrainer` class with a new training method

2. Update the configuration and model selection logic

3. Add appropriate evaluation metrics for the new model

## Business Applications

This framework can be applied to numerous business scenarios:

### Supply Chain Optimization

- Forecast demand fluctuations and identify potential stock-outs

- Optimize inventory levels to reduce carrying costs

- Predict supply chain disruptions before they affect operations

### Financial Risk Management

- Predict market volatility and identify potential financial risks

- Optimize investment strategies based on forecasted trends

- Identify anomalous market behavior for proactive mitigation

### Operations Planning

- Forecast resource requirements for efficient allocation

- Identify potential operational disruptions for proactive planning

- Optimize staffing levels based on predicted demand

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at <http://www.apache.org/licenses/LICENSE-2.0.>

=======
- [API Documentation](docs/api/)
- [User Guides](docs/guides/)
- [Coding Standards](docs/coding_standards/)
- [Project Roadmap](docs/roadmap.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
>>>>>>> 0db33497e3862941e74d9eb69d13a83870c23ad5
