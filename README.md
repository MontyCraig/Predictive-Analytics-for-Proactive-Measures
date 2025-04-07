# Predictive Analytics for Proactive Measures

A comprehensive framework for time series forecasting and disruption prediction in business scenarios.

## Overview

This project provides a robust, enterprise-grade solution for predictive analytics that can be applied to various business scenarios, particularly in supply chain management, demand forecasting, and risk mitigation. By analyzing historical time series data, the system can:

1. Forecast future values with statistical confidence intervals
2. Identify potential disruptions and anomalies in advance
3. Provide actionable insights for proactive business measures

The implementation follows MetaReps coding standards for Python, with strong typing, modular architecture, comprehensive error handling, and thorough documentation.

## Features

- **Data Collection & Preprocessing**: Integrates with Alpha Vantage API (and extendable to other sources) to fetch real-time market data with robust error handling and data validation
- **Exploratory Data Analysis**: Comprehensive time series analysis tools including seasonal decomposition, stationarity testing, and correlation analysis
- **Model Training & Evaluation**: SARIMA model implementation with configurable parameters and extensive evaluation metrics
- **Forecasting & Prediction**: Generate forecasts with confidence intervals and uncertainty quantification
- **Disruption Identification**: Advanced algorithms to detect trend, volatility, level, and uncertainty disruptions

## Project Structure

```
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
│   └── logging_config.py     # Logging setup
├── data_collection_and_preprocessing.py   # Data collection module
├── exploratory_data_analysis.py           # EDA module
├── model_selection_and_training.py        # Model training module
├── forecasting_and_prediction.py          # Forecasting module
├── identifying_potential_disruptions.py   # Disruption detection module
├── main.py                                # Main entry point
├── requirements.txt                       # Dependencies
└── README.md                              # Documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/predictive-analytics-for-proactive-measures.git
   cd Predictive-Analytics-for-Proactive-Measures
   ```

2. Create and activate a conda environment:
   ```
   conda create -n predictive_analytics python=3.9
   conda activate predictive_analytics
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Get an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) for data collection (a free key is available)

## Usage

### Using the Main Script

The simplest way to use the framework is through the main script, which orchestrates the entire workflow:

```bash
python main.py --symbol MSFT --api-key YOUR_API_KEY
```

This will:
1. Collect and preprocess stock data for Microsoft (MSFT)
2. Perform exploratory data analysis with visualizations
3. Train a SARIMA model and evaluate its performance
4. Generate forecasts for the next 30 days
5. Analyze the forecast for potential disruptions

### Command Line Options

```
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
```

### Using Individual Modules

You can also use each module independently for more granular control:

```python
# Example: Just collect and preprocess data
from utils.config import get_config
from data_collection_and_preprocessing import collect_and_preprocess_data

config = get_config()
data = collect_and_preprocess_data(config, symbol="AAPL")
```

## Configuration

The system uses a Pydantic-based configuration management system. You can configure it through:

1. Environment variables
2. A `.env` file in the project root
3. Command-line arguments

Key configuration options:

- `ALPHA_VANTAGE_API_KEY`: Your API key for data collection
- `MODEL_TYPE`: Forecasting model type (default: "sarima")
- `TRAIN_SIZE`: Proportion of data for training (default: 0.8)
- `TEST_SIZE`: Proportion of data for testing (default: 0.2)
- `LOG_LEVEL`: Logging verbosity (default: "INFO")

## Extending the Framework

### Adding New Data Sources

To add a new data source, extend the data collection module:

1. Create a new client class similar to `AlphaVantageClient`
2. Implement the required methods for data fetching and preprocessing
3. Update the configuration system to support the new data source

### Adding New Forecasting Models

To add a new forecasting model:

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

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.


