# Predictive Analytics Examples

This directory contains examples and tutorials demonstrating how to use the Predictive Analytics for Proactive Measures framework. These examples provide practical guidance on implementing forecasting workflows and identifying potential disruptions.

## Available Examples

### `example_workflow.py`

A comprehensive Python script that demonstrates the complete workflow:

1. Data Collection and Preprocessing
2. Exploratory Data Analysis
3. Model Selection and Training
4. Forecasting and Prediction
5. Identifying Potential Disruptions

**Usage:**
```bash
python example_workflow.py
```

This script:
- Fetches stock data for Microsoft (MSFT) from Alpha Vantage
- Performs preprocessing and feature engineering
- Conducts exploratory data analysis with visualizations
- Trains a SARIMA forecasting model
- Generates forecasts for the next 30 days
- Identifies potential disruptions in the forecast
- Saves all outputs (plots, data, models) to appropriate directories

## Running the Examples

### Prerequisites

Before running the examples, ensure you have:

1. Installed all required dependencies (`pip install -e "../.[dev]"`)
2. Created a `.env` file in the project root with your Alpha Vantage API key (`ALPHA_VANTAGE_API_KEY=your_key_here`)
3. Completed the [installation process](../docs/guides/installation.md)

### Environment Setup

The examples are designed to be run from this directory. They automatically set up the necessary Python path to import modules from the parent directory.

## Output

The examples create the following directory structure for outputs:

```
output/
├── eda/
│   └── plots/
├── models/
│   └── plots/
├── forecasts/
│   ├── data/
│   └── plots/
└── disruptions/
    ├── data/
    └── plots/
```

## Customization

You can modify the example scripts to:

- Change the stock symbol for analysis
- Adjust model parameters
- Modify visualization settings
- Change the forecast horizon
- Use different detection thresholds for disruptions

## Additional Resources

For more detailed information, refer to:

- [API Documentation](../docs/api/index.md)
- [User Guides](../docs/guides/basic_usage.md)
- [Feature To-Do Lists](../docs/todos/feature_todos.md) 