# Predictive Analytics for Proactive Measures - ML-Powered Forecasting System

## Overview

Predictive Analytics for Proactive Measures is a comprehensive machine learning framework designed to forecast potential disruptions, identify trends, and enable proactive decision-making across business operations. This system integrates data collection, preprocessing, model training, and forecasting capabilities to transform historical data into actionable insights.

## Purpose

- Forecast potential business disruptions before they occur
- Identify emerging trends and patterns
- Enable proactive rather than reactive decision-making
- Integrate predictions into business strategy
- Reduce operational risks through early warning systems
- Optimize resource allocation based on predictions

## Directory Structure

```text
Predictive-Analytics-for-Proactive-Measures/
├── src/predictive_analytics/      # Installable package (src layout)
│   ├── cli.py                     # Typer CLI behind the `predictive-analytics` command
│   ├── __main__.py                # Enables `python -m predictive_analytics`
│   ├── collection/                # Alpha Vantage client and data preprocessing
│   ├── analysis/                  # Exploratory time-series analysis
│   ├── modeling/                  # Model training and forecasting
│   ├── disruption/                # Disruption / anomaly identification
│   ├── config/                    # Pydantic v2 settings and logging setup
│   ├── utils/                     # General-purpose helpers
│   ├── exceptions.py              # Exception hierarchy
│   └── types.py                   # Shared type definitions
├── tests/                         # pytest suite (CI enforces 100% coverage)
├── docs/                          # API reference, guides, coding standards, roadmap
├── examples/                      # Usage examples
├── .github/workflows/             # Lint, Tests and Security workflows
├── pyproject.toml                 # Package metadata, dependencies, tool configuration
├── .env-example                   # Template for environment configuration
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE                        # Apache License 2.0
└── CLAUDE.md                      # This documentation
```

## Key Components

### Package Modules

| Module | Main public objects | Responsibility |
|---|---|---|
| `collection/client.py` | `AlphaVantageClient`, `TimeSeriesData` | Fetch market time series from Alpha Vantage |
| `collection/preprocessing.py` | `collect_and_preprocess_data`, `preprocess_data` | Clean and prepare collected data |
| `analysis/explorer.py` | `TimeSeriesExplorer`, `load_data` | Exploratory data analysis |
| `modeling/trainer.py` | `ModelTrainer`, `train_and_evaluate_model` | Train and evaluate SARIMA, Prophet, Auto ARIMA and Exponential Smoothing models |
| `modeling/forecaster.py` | `TimeSeriesForecaster`, `generate_forecast` | Generate forecasts with confidence bounds from saved models |
| `disruption/analyzer.py` | `DisruptionAnalyzer`, `analyze_disruptions` | Identify potential disruptions in forecasts |
| `config/settings.py` | `AppConfig`, `get_config`, per-model config classes | Pydantic v2 settings from environment variables / `.env` |
| `cli.py` | `app` | Command-line pipeline orchestration |

The package root re-exports `AlphaVantageClient`, `AppConfig`, `DisruptionAnalyzer`, `ModelTrainer`, `TimeSeriesExplorer` and `TimeSeriesForecaster`.

### Documentation

- `docs/api/` - module-level API reference
- `docs/guides/` - installation, basic usage and Prophet guides
- `docs/coding_standards/` - Python, CLI, packaging and framework coding standards
- `docs/roadmap.md` - development roadmap

## Dependencies

Runtime and development dependencies are declared in `pyproject.toml`; there is no `requirements.txt`. Core runtime libraries: numpy, pandas, scipy, scikit-learn, statsmodels, prophet, pmdarima, matplotlib, seaborn, pydantic and pydantic-settings, python-dotenv, requests, typer, rich, tqdm and joblib. The `dev` extra adds the test, lint and type-checking tools used in CI.

### System Requirements
- Python 3.10+ (`requires-python = ">=3.10"`)
- An Alpha Vantage API key (`ALPHA_VANTAGE_API_KEY`), required by the configuration even when stages are skipped
- 8GB+ RAM recommended for model training

## Installation

```bash
git clone https://github.com/MontyCraig/Predictive-Analytics-for-Proactive-Measures.git
cd Predictive-Analytics-for-Proactive-Measures

# Inside an isolated Python 3.10+ environment
pip install -e ".[dev]"

# Verify the CLI is installed
predictive-analytics --help
```

## Usage

### Command-Line Usage

The CLI exposes a single pipeline command, invoked directly as `predictive-analytics` (there is no `run` subcommand):

```bash
# Full pipeline for one symbol; the API key is read from ALPHA_VANTAGE_API_KEY or a .env file
export ALPHA_VANTAGE_API_KEY=YOUR_API_KEY
predictive-analytics --symbol MSFT

# Skip individual stages
predictive-analytics --symbol AAPL --skip-collection --skip-eda

# Custom forecast horizon (1-365) and output directory (default ./output/<SYMBOL>)
predictive-analytics --symbol GOOG --forecast-steps 60 --output-dir ./results

# Equivalent module invocation
python -m predictive_analytics --symbol MSFT
```

Other options: `--env-file` (load a specific `.env`), `--skip-training`, `--skip-forecasting`, `--skip-disruptions`, `--show-plots`.

Note: `--api-key` is accepted but is not currently passed to the pipeline configuration, so set `ALPHA_VANTAGE_API_KEY` (or use `--env-file`) instead.

### Python API

```python
from predictive_analytics import ModelTrainer
from predictive_analytics.collection.preprocessing import collect_and_preprocess_data
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.forecaster import generate_forecast

config = get_config()  # environment variables / .env

data = collect_and_preprocess_data(config, symbol="MSFT")

trainer = ModelTrainer(config, data=data, target_column="close")
trainer.split_data()
trainer.train_sarima_model()
metrics = trainer.evaluate_model()  # mae, mape, mse, r2, rmse
trainer.save_model("models/sarima_msft.pkl")

# Without model_path, the newest saved <model_type>_*.pkl in the model directory is used
forecast_df = generate_forecast(
    config, model_path="models/sarima_msft.pkl", steps=30, target_column="close"
)  # columns: forecast, lower_bound, upper_bound
```

## Integration Use Cases

### Cluster-wide Predictive Analytics

#### Infrastructure Predictions
- Server failure prediction based on hardware metrics
- Disk failure forecasting using SMART data
- Network congestion prediction
- Resource utilization forecasting

#### Security Predictions
- Anomaly detection for security threats
- Attack pattern prediction
- Vulnerability exploitation forecasting
- Access pattern analysis

#### Operational Predictions
- Workload forecasting for capacity planning
- Maintenance window optimization
- Performance degradation prediction
- Cost forecasting for cloud/hardware resources

### Data Sources
- System logs from all cluster servers
- Performance metrics (CPU, RAM, disk, network)
- Security logs (Wazuh, fail2ban)
- Application logs
- User activity data
- External data (weather, market trends, etc.)

### Integration Points
- **Monitoring Systems**: Feed predictions to monitoring dashboards
- **Alerting**: Generate proactive alerts for predicted issues
- **Automation**: Trigger preventive actions based on predictions
- **Reporting**: Regular prediction reports for stakeholders

## Model Selection Guide

General modelling guidance. The package itself implements SARIMA, Prophet, Auto ARIMA and Exponential Smoothing (`ModelTrainer.train_*_model`); the other model families below are not provided.

### Time Series Models
- **ARIMA**: Seasonal patterns, stationary data
- **Prophet**: Strong seasonality, holidays, trend changes
- **LSTM**: Complex temporal dependencies, non-linear patterns

### Classification Models
- **Random Forest**: Feature importance, non-linear relationships
- **XGBoost**: High performance, handles missing data
- **Neural Networks**: Complex patterns, large datasets

### Regression Models
- **Linear Regression**: Simple relationships, interpretability
- **Ridge/Lasso**: Regularization, feature selection
- **SVR**: Non-linear relationships, robust to outliers

### Ensemble Methods
- **Stacking**: Combine multiple model types
- **Bagging**: Reduce variance, improve stability
- **Boosting**: Improve weak learners iteratively

## Common Commands

### Pipeline
```bash
predictive-analytics --symbol MSFT                       # full pipeline
predictive-analytics --symbol MSFT --skip-collection     # reuse previously collected data
predictive-analytics --symbol MSFT --forecast-steps 90   # longer forecast horizon
```

### Development (mirrors the CI workflows)
```bash
black --check src/ tests/
isort --check-only src/ tests/
flake8 --max-line-length=99 --extend-ignore=E203,W503 src/ tests/
mypy src/
pylint src/predictive_analytics/   # CI disables some checks; see .github/workflows/lint.yml
pytest --cov-fail-under=100
bandit -r src/ -c pyproject.toml
```

## Troubleshooting

### Issue: Poor Model Performance
**Solutions**:
- Collect more training data
- Engineer better features
- Try different models
- Tune hyperparameters
- Check for data leakage

### Issue: Overfitting
**Solutions**:
- Increase regularization
- Reduce model complexity
- Use cross-validation
- Collect more diverse data
- Apply early stopping

### Issue: Long Training Times
**Solutions**:
```python
# Use parallel processing
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)  # Use all cores

# Reduce data size for prototyping
train_sample = train_data.sample(frac=0.1)

# Use incremental learning (classifiers need the full label set on the first call)
import numpy as np
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.partial_fit(X_batch, y_batch, classes=np.unique(y_all))
```

## Security Considerations

- Protect sensitive training data
- Validate input data to prevent poisoning
- Secure model files from tampering
- Audit predictions for bias
- Monitor for adversarial attacks

## Maintenance

### Regular Tasks
- Retrain models with new data (monthly)
- Monitor prediction accuracy (daily)
- Update features as needed (quarterly)
- Evaluate model drift (weekly)
- Archive old models (monthly)

### Model Lifecycle
1. **Development**: Create and test models
2. **Deployment**: Put models into production
3. **Monitoring**: Track performance metrics
4. **Retraining**: Update with new data
5. **Retirement**: Replace outdated models

---

*Last Updated: September 30, 2026*
*Status: Active development and deployment*
