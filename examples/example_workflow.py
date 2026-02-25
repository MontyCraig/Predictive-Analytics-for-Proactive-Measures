#!/usr/bin/env python3
"""Example workflow for the Predictive Analytics for Proactive Measures framework.

This script demonstrates the complete workflow:
1. Data Collection and Preprocessing
2. Exploratory Data Analysis
3. Model Selection and Training
4. Forecasting and Prediction
5. Identifying Potential Disruptions
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

# Set up plotting
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

# Load environment variables from the .env file
load_dotenv()

# Check if API key is available
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if not api_key:
    api_key = input("Enter your Alpha Vantage API key: ")
    os.environ["ALPHA_VANTAGE_API_KEY"] = api_key

# Import package modules
from predictive_analytics.analysis.explorer import TimeSeriesExplorer
from predictive_analytics.collection.preprocessing import collect_and_preprocess_data
from predictive_analytics.config.settings import get_config
from predictive_analytics.disruption.analyzer import DisruptionAnalyzer
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster
from predictive_analytics.modeling.trainer import ModelTrainer


def main() -> None:
    """Run the complete workflow."""
    # Load configuration
    config = get_config()

    print("=" * 80)
    print("PREDICTIVE ANALYTICS FOR PROACTIVE MEASURES")
    print("=" * 80)
    print(f"  Model type: {config.model.model_type}")
    print(f"  Train/Test split: {config.model.train_size}/{config.model.test_size}")
    print(f"  Output directory: {config.output_dir}")
    print()

    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Data Collection and Preprocessing
    print("=" * 80)
    print("STEP 1: DATA COLLECTION AND PREPROCESSING")
    print("=" * 80)

    symbol = "MSFT"

    try:
        print(f"Collecting and preprocessing data for {symbol}...")
        data = collect_and_preprocess_data(config, symbol=symbol, save=True)
        print(f"Successfully collected and preprocessed data for {symbol}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Features: {', '.join(data.columns)}")
    except Exception as exc:
        print(f"Error collecting data: {exc}")
        print("Trying to load from previously saved data...")
        data_path = Path(f"data/preprocessed/{symbol}_preprocessed.csv")
        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"Successfully loaded data from {data_path}")
        else:
            print(f"No preprocessed data found for {symbol}.")
            return

    # Step 2: Exploratory Data Analysis
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    explorer = TimeSeriesExplorer(data, target_column="close")

    print("Testing for stationarity...")
    is_stationary, p_value, critical_values = explorer.test_stationarity()
    print(f"Stationary: {is_stationary}, P-value: {p_value:.4f}")

    plots_dir = Path("output/eda/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    explorer.plot_time_series()
    plt.savefig(plots_dir / f"{symbol}_time_series.png")

    explorer.plot_rolling_statistics(window=20)
    plt.savefig(plots_dir / f"{symbol}_rolling_statistics.png")

    explorer.plot_seasonal_decomposition(period=30, model="additive")
    plt.savefig(plots_dir / f"{symbol}_seasonal_decomposition.png")

    explorer.plot_acf_pacf(lags=40)
    plt.savefig(plots_dir / f"{symbol}_acf_pacf.png")

    print(f"EDA plots saved to {plots_dir}")

    # Step 3: Model Selection and Training
    print("\n" + "=" * 80)
    print("STEP 3: MODEL SELECTION AND TRAINING")
    print("=" * 80)

    trainer = ModelTrainer(config, data, target_column="close")
    train_data, test_data = trainer.split_data()
    print(f"Training: {train_data.shape}, Test: {test_data.shape}")

    trainer.train_sarima_model(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 5),
        exog_columns=["lag_1", "lag_5", "rolling_mean_5"],
    )

    metrics = trainer.evaluate_model(exog_columns=["lag_1", "lag_5", "rolling_mean_5"])
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    model_path = Path(f"models/sarima_{symbol}_{timestamp}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_path)

    # Step 4: Forecasting and Prediction
    print("\n" + "=" * 80)
    print("STEP 4: FORECASTING AND PREDICTION")
    print("=" * 80)

    forecaster = TimeSeriesForecaster(
        config, model_path=model_path, data=data, target_column="close"
    )
    forecast_df = forecaster.forecast(steps=30, confidence_interval=0.95)
    print(forecast_df.head())

    analysis = forecaster.analyze_forecast(forecast_df)
    for key, val in analysis.items():
        print(f"  {key}: {val}")

    # Step 5: Identifying Potential Disruptions
    print("\n" + "=" * 80)
    print("STEP 5: IDENTIFYING POTENTIAL DISRUPTIONS")
    print("=" * 80)

    analyzer = DisruptionAnalyzer(
        config, forecast_data=forecast_df, historical_data=data, target_column="close"
    )
    disruption_df = analyzer.identify_all_disruptions()
    report = analyzer.generate_disruption_report(disruption_df)

    print(f"Significant disruptions: {report['significant_disruptions']}")
    print(f"Disruption percentage: {report['disruption_percentage']:.2f}%")

    print("\nWorkflow complete!")


if __name__ == "__main__":
    main()
