# Main Module

## Overview

The Main module serves as the entry point for the Predictive Analytics for Proactive Measures framework. It orchestrates the entire workflow, from data collection to disruption analysis, providing a unified interface for the various components of the system.

## Architecture

The module follows a procedural architecture with these primary components:

1. **Environment Setup**: Functions to initialize required directories and configuration
2. **Workflow Orchestration**: Functions to run the complete predictive analytics pipeline
3. **Command-line Interface**: Argument parsing and job delegation
4. **Configuration Integration**: Functions to manage configuration from the command line

## API Reference

### Functions

#### `setup_environment() -> None`

Set up the environment by creating necessary directories.

#### `create_env_file(api_key: str = None) -> Path`

Create a .env file with API keys.

**Parameters:**
- `api_key` (str, optional): Alpha Vantage API key

**Returns:**
- `Path`: Path to the created .env file

#### `run_full_workflow(config: Config, symbol: str = "MSFT", output_dir: Optional[str] = None, skip_collection: bool = False, skip_eda: bool = False, skip_training: bool = False, skip_forecasting: bool = False, skip_disruptions: bool = False, show_plots: bool = False, forecast_steps: int = 30) -> None`

Run the full predictive analytics workflow.

**Parameters:**
- `config` (Config): Configuration object
- `symbol` (str, optional): Stock symbol to analyze (default: "MSFT")
- `output_dir` (str, optional): Directory to save outputs
- `skip_collection` (bool, optional): Whether to skip data collection (default: False)
- `skip_eda` (bool, optional): Whether to skip exploratory data analysis (default: False)
- `skip_training` (bool, optional): Whether to skip model training (default: False)
- `skip_forecasting` (bool, optional): Whether to skip forecasting (default: False)
- `skip_disruptions` (bool, optional): Whether to skip disruption analysis (default: False)
- `show_plots` (bool, optional): Whether to show plots instead of saving (default: False)
- `forecast_steps` (int, optional): Number of steps to forecast (default: 30)

### Command-line Interface

The main module provides a comprehensive command-line interface with the following arguments:

- `--symbol`: Stock symbol to analyze (default: "MSFT")
- `--api-key`: Alpha Vantage API key
- `--env-file`: Path to .env file with API keys
- `--output-dir`: Directory to save outputs
- `--skip-collection`: Skip data collection
- `--skip-eda`: Skip exploratory data analysis
- `--skip-training`: Skip model training
- `--skip-forecasting`: Skip forecasting
- `--skip-disruptions`: Skip disruption analysis
- `--show-plots`: Show plots instead of saving
- `--forecast-steps`: Number of steps to forecast (default: 30)

## Usage Examples

### Basic Usage

Run the complete workflow for the default stock (Microsoft):

```bash
python main.py
```

### Custom Stock Analysis

Analyze a different stock symbol:

```bash
python main.py --symbol AAPL
```

### Skip Certain Steps

Run only specific parts of the workflow:

```bash
python main.py --symbol GOOGL --skip-collection --skip-eda
```

### Custom API Key

Provide an Alpha Vantage API key:

```bash
python main.py --api-key YOUR_API_KEY
```

### Interactive Plots

Show plots interactively instead of saving them:

```bash
python main.py --show-plots
```

### Custom Forecast Horizon

Specify the number of days to forecast:

```bash
python main.py --forecast-steps 60
```

## Implementation Details

### Workflow Orchestration

The main module orchestrates the following steps:

1. **Environment Setup**: Ensuring all necessary directories exist
2. **Configuration Management**: Loading configuration from environment variables or command line
3. **Data Collection and Preprocessing**: Fetching and preparing time series data
4. **Exploratory Data Analysis**: Analyzing temporal patterns and statistical properties
5. **Model Selection and Training**: Training and evaluating forecasting models
6. **Forecasting and Prediction**: Generating future predictions with confidence intervals
7. **Disruption Identification**: Detecting potential anomalies and disruptions

### Output Management

The module manages outputs as follows:

1. **Run-specific Directories**: Creating timestamped directories for each run
2. **Organization**: Maintaining separate subdirectories for raw data, preprocessed data, models, and analysis results
3. **Reporting**: Generating summary reports for model performance and disruption analysis

## Integration Points

The main module integrates with:
- Command-line argument parser (argparse)
- Configuration management system
- All other modules in the framework
- File system for environment setup and output management
- Logging system for operational insights

## Areas for Enhancement

1. **Web Interface**: Add a web-based user interface for easier interaction
2. **Scheduled Execution**: Add support for automated scheduled runs
3. **Parallel Processing**: Implement parallel execution of independent steps
4. **Workflow Configuration**: Allow defining custom workflows via configuration files
5. **Progress Reporting**: Enhance progress reporting during execution
6. **Error Recovery**: Add mechanisms to resume failed workflows 