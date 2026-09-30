# Predictive Analytics for Proactive Measures

Enterprise-grade time series forecasting and disruption prediction framework.

## Overview

A modular Python package for collecting market data, training forecasting models, and identifying potential disruptions before they impact business operations. Supports SARIMA, Prophet, Auto ARIMA, and Exponential Smoothing models with fully typed Pydantic v2 configuration.

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

## Installation

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

### CLI Usage

```bash
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

# Load configuration from environment / .env file
config = get_config()

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

## Configuration

Configuration is managed through Pydantic v2 models with environment variable support:

```bash
# Required
export ALPHA_VANTAGE_API_KEY=your_key_here

# Optional (with defaults)
export MODEL_TYPE=sarima          # sarima | prophet | auto_arima | exp_smoothing
export TRAIN_SIZE=0.8
export TEST_SIZE=0.2
export LOG_LEVEL=INFO
```

Or use a `.env` file (see `.env-example` for all available options).

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (requires 100% coverage)
pytest

# Code quality
black src/ tests/
isort src/ tests/
mypy --strict src/
flake8 src/ tests/
bandit -r src/ -c pyproject.toml
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Documentation

- [API Documentation](docs/api/)
- [User Guides](docs/guides/)
- [Coding Standards](docs/coding_standards/)
- [Project Roadmap](docs/roadmap.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
