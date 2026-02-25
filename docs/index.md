# Predictive Analytics for Proactive Measures

Enterprise-grade time series forecasting and disruption prediction framework.

## Architecture

The package follows a modular pipeline architecture:

1. **Collection** -- Fetch and validate market data from Alpha Vantage
2. **Analysis** -- Explore temporal patterns, seasonality, and stationarity
3. **Modeling** -- Train and evaluate SARIMA, Prophet, Auto ARIMA, or Exponential Smoothing models
4. **Forecasting** -- Generate predictions with confidence intervals
5. **Disruption Detection** -- Identify trend, volatility, level, and uncertainty disruptions

## Package Reference

| Module | Class / Function | Description |
|--------|------------------|-------------|
| `collection.client` | `AlphaVantageClient` | API client with rate-limit handling |
| `collection.preprocessing` | `preprocess_data`, `collect_and_preprocess_data` | Data cleaning and feature engineering |
| `analysis.explorer` | `TimeSeriesExplorer` | 11 EDA methods (decomposition, ACF, stationarity) |
| `modeling.trainer` | `ModelTrainer` | Train 4 model types with evaluation metrics |
| `modeling.forecaster` | `TimeSeriesForecaster` | Forecast with confidence intervals |
| `disruption.analyzer` | `DisruptionAnalyzer` | Detect 4 categories of disruption |
| `config.settings` | `AppConfig`, `get_config` | Pydantic v2 configuration management |
| `exceptions` | `PredictiveAnalyticsError` hierarchy | Domain-specific exception classes |
| `types` | `SARIMAOrder`, `MetricsDict`, `DisruptionReport` | Shared type aliases |

## API Documentation

- [Data Collection and Preprocessing](./api/data_collection_and_preprocessing.md)
- [Exploratory Data Analysis](./api/exploratory_data_analysis.md)
- [Model Selection and Training](./api/model_selection_and_training.md)
- [Forecasting and Prediction](./api/forecasting_and_prediction.md)
- [Disruption Analysis](./api/identifying_potential_disruptions.md)
- [Utilities](./api/utils.md)

## Guides

- [Installation Guide](./guides/installation.md)
- [Basic Usage](./guides/basic_usage.md)
- [Advanced Configuration](./guides/advanced_configuration.md)
- [Adding Data Sources](./guides/adding_data_sources.md)
- [Adding Models](./guides/adding_models.md)

## Standards

- [Python Coding Standards](./coding_standards/python_coding_standards.md)
- [Pydantic Standards](./coding_standards/pydantic_coding_standards.md)
- [Testing Standards](./coding_standards/testing_standards.md)
- [Performance Standards](./coding_standards/performance_standards.md)

## Project

- [Roadmap](./roadmap.md)
- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md)
- [Security](../SECURITY.md)

## License

Apache License 2.0.
