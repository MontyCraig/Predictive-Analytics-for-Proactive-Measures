#!/usr/bin/env python3
"""Example of using Facebook Prophet for time series forecasting.

This script demonstrates how to:
1. Collect and preprocess stock market data
2. Train a Prophet model
3. Evaluate the model
4. Generate forecasts
5. Visualize results
6. Identify potential disruptions
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up plotting
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Import package modules
from predictive_analytics.collection.preprocessing import collect_and_preprocess_data
from predictive_analytics.config.logging import setup_logging
from predictive_analytics.config.settings import get_config
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster
from predictive_analytics.modeling.trainer import ModelTrainer

logger = setup_logging()


def run_prophet_example() -> None:
    """Run a complete example of Prophet forecasting."""
    print("\n" + "=" * 80)
    print("FACEBOOK PROPHET FORECASTING EXAMPLE")
    print("=" * 80)

    # 1. Load configuration
    config = get_config()
    config.model.model_type = "prophet"

    # 2. Collect and preprocess data
    symbol = "MSFT"

    try:
        data = collect_and_preprocess_data(config, symbol=symbol, save=True)
        print(f"Successfully collected data for {symbol}")
    except Exception as exc:
        print(f"Error collecting data: {exc}")
        data_path = Path(f"data/preprocessed/{symbol}_preprocessed.csv")
        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            print("No data available.")
            return

    # 3. Create output directory
    output_dir = Path("output/examples/prophet")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Train a Prophet model
    trainer = ModelTrainer(config, data, target_column="close")
    train_data, test_data = trainer.split_data(test_size=0.2)
    print(f"Training: {len(train_data)} records, Test: {len(test_data)} records")

    trainer.train_prophet_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        exog_columns=["volume", "day_of_week", "month"],
    )

    # 5. Evaluate the model
    metrics = trainer.evaluate_model(exog_columns=["volume", "day_of_week", "month"])
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    eval_plot_path = output_dir / "prophet_evaluation.png"
    trainer.plot_results(save_path=eval_plot_path)

    # 6. Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"prophet_{symbol}_{timestamp}.pkl"
    trainer.save_model(model_path)

    # 7. Generate forecasts
    forecaster = TimeSeriesForecaster(config, model_path=model_path, data=data)

    forecast_days = 60
    forecast_df = forecaster.forecast(steps=forecast_days)
    print(f"\nForecast Summary (first 5 days):")
    print(forecast_df.head())

    forecast_plot_path = output_dir / "prophet_forecast.png"
    forecaster.plot_forecast(forecast_df, historical_periods=90, save_path=forecast_plot_path)

    # 8. Analyse the forecast
    analysis = forecaster.analyze_forecast(forecast_df)
    for metric, value in analysis.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # 9. Detect anomalies
    disruptions_df = forecaster.detect_anomalies(forecast_df, threshold=2.0)
    uncertain_periods = disruptions_df[disruptions_df["is_uncertain"]].sort_values(
        by="range_z_score", ascending=False
    )
    print(f"\nDetected {len(uncertain_periods)} periods with high uncertainty")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    run_prophet_example()
