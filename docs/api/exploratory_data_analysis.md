# Exploratory Data Analysis Module

## Overview

The Exploratory Data Analysis (EDA) module provides a comprehensive suite of tools for analyzing time series data. It includes visualization capabilities, statistical tests, and decomposition methods to help understand the underlying patterns, trends, seasonality, and anomalies in the data.

## Architecture

The module is designed around the `TimeSeriesExplorer` class, which encapsulates various analysis methods and visualization tools. The architecture follows a cohesive design where:

1. **Core Explorer Class**: Centralizes all analytical capabilities
2. **Visualization Methods**: Generate insightful plots for different aspects of the data
3. **Statistical Analysis**: Implements tests and metrics for time series properties
4. **Helper Functions**: Support loading and transforming data

## API Reference

### `TimeSeriesExplorer` (Class)

Explorer for time series data with visualization capabilities.

#### `__init__(data: pd.DataFrame, target_column: str = "close") -> None`

Initialize the explorer with time series data.

**Parameters:**
- `data` (pd.DataFrame): Time series data
- `target_column` (str, optional): Target column to analyze (default: "close")

#### `plot_time_series(columns: Optional[List[str]] = None) -> None`

Plot the time series data.

**Parameters:**
- `columns` (List[str], optional): List of columns to plot (defaults to target_column)

#### `plot_seasonal_decomposition(period: int = 30, model: str = "additive") -> None`

Plot seasonal decomposition of the time series.

**Parameters:**
- `period` (int, optional): Period for seasonal component (default: 30)
- `model` (str, optional): Type of decomposition ('additive' or 'multiplicative') (default: "additive")

#### `plot_acf_pacf(lags: int = 40) -> None`

Plot autocorrelation and partial autocorrelation functions.

**Parameters:**
- `lags` (int, optional): Number of lags to include (default: 40)

#### `test_stationarity() -> Tuple[float, float, dict]`

Test stationarity of the time series using the Augmented Dickey-Fuller test.

**Returns:**
- `Tuple[float, float, dict]`: Tuple of test statistic, p-value, and critical values

#### `plot_rolling_statistics(window: int = 20) -> None`

Plot rolling mean and standard deviation.

**Parameters:**
- `window` (int, optional): Window size for rolling statistics (default: 20)

#### `plot_distribution() -> None`

Plot the distribution of the target column with histogram and Q-Q plot.

#### `plot_box_plots(by: str = "month") -> None`

Plot box plots by time period.

**Parameters:**
- `by` (str, optional): Time period to group by ('month', 'quarter', 'year', 'day_of_week') (default: "month")

#### `plot_heatmap(columns: Optional[List[str]] = None) -> None`

Plot correlation heatmap.

**Parameters:**
- `columns` (List[str], optional): List of columns to include (defaults to all numeric columns)

#### `plot_lag_scatter(max_lag: int = 10) -> None`

Plot scatter plots between the series and its lags.

**Parameters:**
- `max_lag` (int, optional): Maximum lag to include (default: 10)

#### `run_full_analysis(output_dir: Optional[Path] = None) -> None`

Run a complete exploratory analysis with all plots.

**Parameters:**
- `output_dir` (Path, optional): Directory to save plots (if None, plots are displayed)

### `load_data(file_path: Union[str, Path]) -> pd.DataFrame`

Load data from a CSV file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the CSV file

**Returns:**
- `pd.DataFrame`: DataFrame with time series data

**Raises:**
- `FileNotFoundError`: If the file doesn't exist
- `Exception`: For other errors during loading

## Usage Examples

### Basic Usage

```python
from predictive_analytics.analysis.explorer import TimeSeriesExplorer
import pandas as pd

# Load data
data = pd.read_csv('data/msft_preprocessed.csv', index_col=0, parse_dates=True)

# Create explorer
explorer = TimeSeriesExplorer(data)

# Run complete analysis
explorer.run_full_analysis()
```

### Custom Analysis

```python
from predictive_analytics.analysis.explorer import TimeSeriesExplorer
import pandas as pd

# Load data
data = pd.read_csv('data/aapl_preprocessed.csv', index_col=0, parse_dates=True)

# Create explorer with custom target column
explorer = TimeSeriesExplorer(data, target_column='high')

# Test stationarity
explorer.test_stationarity()

# Plot seasonal decomposition with custom period
explorer.plot_seasonal_decomposition(period=7, model='multiplicative')

# Plot correlation heatmap for specific columns
explorer.plot_heatmap(columns=['open', 'high', 'low', 'close', 'volume'])
```

### Saving Plots

```python
from predictive_analytics.analysis.explorer import TimeSeriesExplorer
from pathlib import Path
import pandas as pd

# Load data
data = pd.read_csv('data/tsla_preprocessed.csv', index_col=0, parse_dates=True)

# Create explorer
explorer = TimeSeriesExplorer(data)

# Save all plots to a directory
output_dir = Path('output/eda_results')
explorer.run_full_analysis(output_dir)
```

## Implementation Details

### Visualization Techniques

1. **Time Series Plots**: Basic visualization of the time series data
2. **Seasonal Decomposition**: Separation of trend, seasonality, and residual components
3. **Autocorrelation**: ACF and PACF plots for identifying autoregressive and moving average components
4. **Rolling Statistics**: Visualization of rolling mean and standard deviation to check for stationarity
5. **Distribution Analysis**: Histogram and Q-Q plots to assess normality
6. **Box Plots**: Distribution visualization by time periods
7. **Correlation Analysis**: Heatmap of correlations between variables
8. **Lag Scatter Plots**: Visualization of relationships between the series and its lagged values

### Statistical Tests

1. **Augmented Dickey-Fuller Test**: Test for stationarity of the time series
2. **Normality Assessment**: Through histogram and Q-Q plot visualization

## Integration Points

The module integrates with:
- Configuration management system
- Logging system for operational insights
- Visualization libraries (Matplotlib, Seaborn)
- Statistical libraries (StatsModels, SciPy)

## Areas for Enhancement

1. **Additional Statistical Tests**: Implement more comprehensive tests for time series properties
2. **Interactive Visualizations**: Add support for interactive plots using libraries like Plotly
3. **Multivariate Analysis**: Extend analysis capabilities for multivariate time series
4. **Anomaly Detection**: Add specialized visualizations for anomaly detection
5. **Report Generation**: Implement automatic report generation with insights and recommendations 