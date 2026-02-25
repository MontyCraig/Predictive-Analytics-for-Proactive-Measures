# Model Selection and Training Module

## Overview

The Model Selection and Training module provides a comprehensive framework for selecting, training, and evaluating time series forecasting models. The module currently focuses on SARIMA (Seasonal AutoRegressive Integrated Moving Average) models but is designed to be extensible to other model types.

## Architecture

The module follows a clean, object-oriented architecture with these primary components:

1. **Model Trainer**: Core class that handles model training and evaluation
2. **Configuration Integration**: Uses Pydantic models for robust configuration
3. **Evaluation Metrics**: Implements comprehensive metrics for model assessment
4. **Persistence Layer**: Handles model serialization and loading

## API Reference

### `ModelTrainer` (Class)

Trainer for time series forecasting models.

#### `__init__(config: Config, data: Optional[pd.DataFrame] = None, target_column: str = "close") -> None`

Initialize the model trainer.

**Parameters:**
- `config` (Config): Application configuration
- `data` (pd.DataFrame, optional): Time series data
- `target_column` (str, optional): Target column to predict (default: "close")

#### `load_data(file_path: Union[str, Path]) -> None`

Load data from a CSV file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the CSV file

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `Exception`: For other errors during loading

#### `split_data(test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]`

Split data into training and test sets.

**Parameters:**
- `test_size` (float, optional): Proportion of data to use for testing

**Returns:**
- `Tuple[pd.DataFrame, pd.DataFrame]`: Tuple of (train_data, test_data)

**Raises:**
- `ValueError`: If data is not loaded

#### `train_sarima_model(order: Optional[Tuple[int, int, int]] = None, seasonal_order: Optional[Tuple[int, int, int, int]] = None, exog_columns: Optional[List[str]] = None) -> None`

Train a SARIMA model.

**Parameters:**
- `order` (Tuple[int, int, int], optional): SARIMA order (p, d, q)
- `seasonal_order` (Tuple[int, int, int, int], optional): SARIMA seasonal order (P, D, Q, s)
- `exog_columns` (List[str], optional): List of exogenous variables to include

**Raises:**
- `ValueError`: If data is not split or exogenous columns are not found

#### `evaluate_model(exog_columns: Optional[List[str]] = None) -> Dict[str, float]`

Evaluate the trained model on test data.

**Parameters:**
- `exog_columns` (List[str], optional): List of exogenous variables to include

**Returns:**
- `Dict[str, float]`: Dictionary of evaluation metrics

**Raises:**
- `ValueError`: If model is not trained or exogenous columns are not found

#### `save_model(file_path: Union[str, Path]) -> None`

Save the trained model to a file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to save the model

**Raises:**
- `ValueError`: If model is not trained
- `Exception`: For other errors during saving

#### `load_model(file_path: Union[str, Path]) -> None`

Load a trained model from a file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the model file

**Raises:**
- `FileNotFoundError`: If the model file doesn't exist
- `Exception`: For other errors during loading

#### `plot_results(save_path: Optional[Path] = None) -> None`

Plot actual vs predicted values.

**Parameters:**
- `save_path` (Path, optional): Path to save the plot

**Raises:**
- `ValueError`: If model is not evaluated

### `train_and_evaluate_model(config: Config, data: Optional[pd.DataFrame] = None, file_path: Optional[Union[str, Path]] = None, target_column: str = "close", exog_columns: Optional[List[str]] = None, save_model: bool = True, save_plot: bool = True) -> Dict[str, float]`

Train and evaluate a time series model.

**Parameters:**
- `config` (Config): Application configuration
- `data` (pd.DataFrame, optional): Time series data
- `file_path` (Union[str, Path], optional): Path to data file
- `target_column` (str, optional): Target column to predict (default: "close")
- `exog_columns` (List[str], optional): List of exogenous variables to include
- `save_model` (bool, optional): Whether to save the trained model (default: True)
- `save_plot` (bool, optional): Whether to save the results plot (default: True)

**Returns:**
- `Dict[str, float]`: Dictionary of evaluation metrics

## Usage Examples

### Basic Usage

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.trainer import train_and_evaluate_model

# Load configuration
config = get_config()

# Train and evaluate model
metrics = train_and_evaluate_model(
    config,
    target_column="close",
    exog_columns=["lag_1", "lag_5", "rolling_mean_5"]
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
```

### Custom Training with Existing Data

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.trainer import ModelTrainer
import pandas as pd

# Load configuration
config = get_config()

# Load data
data = pd.read_csv('data/preprocessed/AAPL_preprocessed.csv', index_col=0, parse_dates=True)

# Create trainer
trainer = ModelTrainer(config, data, target_column="close")

# Split data
trainer.split_data(test_size=0.3)

# Train model with custom parameters
trainer.train_sarima_model(
    order=(2, 1, 2),
    seasonal_order=(1, 0, 1, 5),
    exog_columns=["lag_1", "rolling_mean_5", "month"]
)

# Evaluate model
metrics = trainer.evaluate_model(exog_columns=["lag_1", "rolling_mean_5", "month"])

# Save model
trainer.save_model('models/custom_sarima_model.pkl')

# Plot results
trainer.plot_results()
```

### Loading a Pre-trained Model

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.trainer import ModelTrainer
import pandas as pd

# Load configuration
config = get_config()

# Create trainer
trainer = ModelTrainer(config)

# Load model
trainer.load_model('models/sarima_20240101_120000.pkl')

# Load new data for prediction
data = pd.read_csv('data/new_data.csv', index_col=0, parse_dates=True)
trainer.data = data

# Split data
trainer.split_data()

# Evaluate on new data
metrics = trainer.evaluate_model(exog_columns=["lag_1", "lag_5", "rolling_mean_5"])

# Plot results
trainer.plot_results(save_path='output/plots/new_data_evaluation.png')
```

## Implementation Details

### Model Types

Currently, the module primarily supports SARIMA models, which are suitable for time series data with seasonal patterns. The implementation leverages the StatsModels library's SARIMAX class.

### Evaluation Metrics

The module calculates and reports multiple evaluation metrics:

1. **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
2. **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
3. **RMSE (Root Mean Squared Error)**: Square root of MSE, providing an error measure in the same units as the target
4. **R² (R-squared)**: Proportion of variance in the target that is predictable from the features
5. **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values

### Model Persistence

Models are serialized using Python's pickle module, with proper error handling and logging. The serialized models include all necessary parameters for prediction, including transformation parameters.

## Integration Points

The module integrates with:
- Configuration management system for model parameters
- Logging system for operation insights
- Data preprocessing module for input data
- File system for model persistence
- Visualization system for result analysis

## Areas for Enhancement

1. **Additional Model Types**: Extend support to other forecasting models such as Prophet, LSTM, and Exponential Smoothing
2. **Hyperparameter Optimization**: Implement automated hyperparameter tuning using grid search or Bayesian optimization
3. **Cross-Validation**: Add time series cross-validation for more robust model evaluation
4. **Model Comparison**: Implement functionality to compare multiple model types and configurations
5. **Feature Importance**: Add analysis of feature importance and model interpretability
6. **Ensemble Methods**: Implement ensemble techniques to combine predictions from multiple models 