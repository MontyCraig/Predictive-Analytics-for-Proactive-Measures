# Identifying Potential Disruptions Module

## Overview

The Identifying Potential Disruptions module provides a comprehensive framework for detecting and analyzing potential anomalies, disruptions, and unusual patterns in time series forecasts. It implements multiple detection algorithms to identify different types of disruptions, allowing businesses to take proactive measures.

## Architecture

The module follows a clean, object-oriented architecture with these primary components:

1. **Disruption Analyzer**: Core class for detecting various types of disruptions
2. **Detection Algorithms**: Specialized methods for different types of disruptions
3. **Visualization Tools**: Methods for visualizing disruptions in forecasts
4. **Reporting System**: Functions for generating comprehensive disruption reports

## API Reference

### `DisruptionAnalyzer` (Class)

Analyzer for identifying potential disruptions in time series data.

#### `__init__(config: Config, forecast_data: Optional[pd.DataFrame] = None, historical_data: Optional[pd.DataFrame] = None, target_column: str = "close") -> None`

Initialize the disruption analyzer.

**Parameters:**
- `config` (Config): Application configuration
- `forecast_data` (pd.DataFrame, optional): DataFrame with forecast data
- `historical_data` (pd.DataFrame, optional): DataFrame with historical data
- `target_column` (str, optional): Target column to analyze (default: "close")

#### `load_forecast(file_path: Union[str, Path]) -> None`

Load forecast data from a CSV file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the forecast CSV file

**Raises:**
- `FileNotFoundError`: If the forecast file doesn't exist
- `Exception`: For other errors during loading

#### `load_historical_data(file_path: Union[str, Path]) -> None`

Load historical data from a CSV file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the historical data CSV file

**Raises:**
- `FileNotFoundError`: If the historical data file doesn't exist
- `Exception`: For other errors during loading

#### `identify_trend_disruptions(window_size: int = 5, threshold: float = 2.0) -> pd.DataFrame`

Identify disruptions in the trend of the forecast.

**Parameters:**
- `window_size` (int, optional): Size of the rolling window for slope calculation (default: 5)
- `threshold` (float, optional): Z-score threshold for disruption identification (default: 2.0)

**Returns:**
- `pd.DataFrame`: DataFrame with trend disruptions flagged

**Raises:**
- `ValueError`: If forecast data is not loaded

#### `identify_volatility_disruptions(window_size: int = 10, threshold: float = 2.0) -> pd.DataFrame`

Identify disruptions in the volatility of the forecast.

**Parameters:**
- `window_size` (int, optional): Size of the rolling window for volatility calculation (default: 10)
- `threshold` (float, optional): Z-score threshold for disruption identification (default: 2.0)

**Returns:**
- `pd.DataFrame`: DataFrame with volatility disruptions flagged

**Raises:**
- `ValueError`: If forecast data is not loaded

#### `identify_level_disruptions(threshold_std: float = 2.0) -> pd.DataFrame`

Identify sudden level shifts in the forecast.

**Parameters:**
- `threshold_std` (float, optional): Number of standard deviations for level shift identification (default: 2.0)

**Returns:**
- `pd.DataFrame`: DataFrame with level shifts flagged

**Raises:**
- `ValueError`: If forecast data is not loaded

#### `identify_uncertainty_disruptions(threshold: float = 2.0) -> pd.DataFrame`

Identify periods with abnormal prediction uncertainty.

**Parameters:**
- `threshold` (float, optional): Z-score threshold for uncertainty identification (default: 2.0)

**Returns:**
- `pd.DataFrame`: DataFrame with uncertainty disruptions flagged

**Raises:**
- `ValueError`: If forecast data is not loaded or doesn't contain prediction intervals

#### `identify_all_disruptions() -> pd.DataFrame`

Identify all types of disruptions in the forecast.

**Returns:**
- `pd.DataFrame`: DataFrame with all disruptions flagged

**Raises:**
- `ValueError`: If forecast data is not loaded

#### `plot_disruptions(disruption_df: pd.DataFrame, historical_periods: int = 60, save_path: Optional[Union[str, Path]] = None) -> None`

Plot forecasts with disruptions highlighted.

**Parameters:**
- `disruption_df` (pd.DataFrame): DataFrame with disruptions flagged
- `historical_periods` (int, optional): Number of historical periods to show (default: 60)
- `save_path` (Union[str, Path], optional): Path to save the plot

#### `generate_disruption_report(disruption_df: pd.DataFrame) -> Dict`

Generate a report summarizing the disruptions.

**Parameters:**
- `disruption_df` (pd.DataFrame): DataFrame with disruptions flagged

**Returns:**
- `Dict`: Dictionary with disruption summary

### `analyze_disruptions(config: Config, forecast_path: Optional[Union[str, Path]] = None, historical_data_path: Optional[Union[str, Path]] = None, target_column: str = "close", generate_new_forecast: bool = False, forecast_steps: int = 30, plot: bool = True, save_results: bool = True) -> Tuple[pd.DataFrame, Dict]`

Analyze potential disruptions in time series forecast.

**Parameters:**
- `config` (Config): Application configuration
- `forecast_path` (Union[str, Path], optional): Path to forecast data file
- `historical_data_path` (Union[str, Path], optional): Path to historical data file
- `target_column` (str, optional): Target column to analyze (default: "close")
- `generate_new_forecast` (bool, optional): Whether to generate a new forecast (default: False)
- `forecast_steps` (int, optional): Number of steps to forecast if generating new forecast (default: 30)
- `plot` (bool, optional): Whether to plot the results (default: True)
- `save_results` (bool, optional): Whether to save the results (default: True)

**Returns:**
- `Tuple[pd.DataFrame, Dict]`: Tuple of (disruption_df, disruption_report)

## Usage Examples

### Basic Usage

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.disruption.analyzer import analyze_disruptions

# Load configuration
config = get_config()

# Analyze disruptions using latest forecast data
disruption_df, report = analyze_disruptions(
    config,
    target_column="close",
    generate_new_forecast=True,
    forecast_steps=30
)

# Print report summary
print(f"Total forecast periods: {report['total_forecast_periods']}")
print(f"Significant disruptions: {report['significant_disruptions']} ({report['disruption_percentage']:.2f}%)")
```

### Custom Analysis with Existing Forecast

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.disruption.analyzer import DisruptionAnalyzer
from pathlib import Path

# Load configuration
config = get_config()

# Create analyzer
analyzer = DisruptionAnalyzer(config)

# Load forecast and historical data
analyzer.load_forecast("output/forecasts/data/forecast_30_steps_20240101_120000.csv")
analyzer.load_historical_data("data/preprocessed/AAPL_preprocessed.csv")

# Identify specific types of disruptions
trend_df = analyzer.identify_trend_disruptions(window_size=7, threshold=2.5)
volatility_df = analyzer.identify_volatility_disruptions(window_size=5, threshold=2.0)

# Count disruptions
trend_disruptions = trend_df["trend_disruption"].sum()
volatility_disruptions = volatility_df["volatility_disruption"].sum()

print(f"Trend disruptions: {trend_disruptions}")
print(f"Volatility disruptions: {volatility_disruptions}")

# Plot trend disruptions
analyzer.plot_disruptions(trend_df, save_path="output/disruptions/trend_disruptions.png")
```

### Comprehensive Disruption Analysis

```python
from predictive_analytics.config.settings import get_config
from predictive_analytics.disruption.analyzer import DisruptionAnalyzer
import pandas as pd
from pathlib import Path

# Load configuration
config = get_config()

# Create analyzer
analyzer = DisruptionAnalyzer(config, target_column="close")

# Load forecast and historical data
analyzer.load_forecast("output/forecasts/data/forecast_60_steps_20240101_120000.csv")
analyzer.load_historical_data("data/preprocessed/MSFT_preprocessed.csv")

# Identify all disruptions
disruption_df = analyzer.identify_all_disruptions()

# Generate report
report = analyzer.generate_disruption_report(disruption_df)

# Save reports to files
output_dir = Path("output/disruptions")
output_dir.mkdir(parents=True, exist_ok=True)

# Save detailed disruption data
disruption_df.to_csv(output_dir / "all_disruptions.csv")

# Plot disruptions
analyzer.plot_disruptions(
    disruption_df,
    historical_periods=90,
    save_path=output_dir / "disruption_analysis.png"
)

# Output significant disruption dates
print("Significant disruption dates:")
for date in report["significant_disruption_dates"]:
    print(f"  - {date.strftime('%Y-%m-%d')}")
```

## Implementation Details

### Disruption Types

The module identifies four primary types of disruptions:

1. **Trend Disruptions**: Unusual changes in the slope of the forecast
2. **Volatility Disruptions**: Abnormal changes in the variability of the forecast
3. **Level Disruptions**: Sudden jumps or drops in the forecast value
4. **Uncertainty Disruptions**: Periods with unusually wide or narrow prediction intervals

### Detection Algorithms

Different statistical approaches are used for each disruption type:

1. **Trend Detection**: Calculates rolling slopes and identifies abnormal slopes using z-scores
2. **Volatility Detection**: Measures rolling standard deviation and flags unusual volatility using z-scores
3. **Level Detection**: Identifies sudden changes between consecutive forecast points
4. **Uncertainty Detection**: Analyzes prediction interval widths to find periods of unusually high or low certainty

### Significance Assessment

The module combines different disruption types to assess overall significance:

1. **Total Disruptions**: Count of different disruption types for each time point
2. **Significant Disruptions**: Time points with multiple types of disruptions (2 or more)
3. **Disruption Percentage**: Proportion of forecast periods with significant disruptions

## Integration Points

The module integrates with:
- Configuration management system
- Forecasting module for generating new forecasts
- File system for storing disruption analysis results
- Visualization system for creating disruption plots
- Logging system for operational insights

## Areas for Enhancement

1. **Machine Learning Detection**: Implement supervised or unsupervised learning approaches for disruption detection
2. **Alert System**: Add functionality to send alerts when significant disruptions are detected
3. **Root Cause Analysis**: Develop methods to help identify the underlying causes of disruptions
4. **Impact Assessment**: Add functionality to estimate the business impact of identified disruptions
5. **Interactive Visualization**: Create interactive dashboards for exploring disruption patterns
6. **Early Warning System**: Improve early detection capabilities for proactive measures 