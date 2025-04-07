#!/usr/bin/env python3
"""
Example of using Facebook Prophet for time series forecasting.

This script demonstrates how to:
1. Collect and preprocess stock market data
2. Train a Prophet model
3. Evaluate the model
4. Generate forecasts
5. Visualize results
6. Identify potential disruptions
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import modules
sys.path.append('..')

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Import our modules
from utils.config import get_config
from utils.logging_config import setup_logging
from data_collection_and_preprocessing import collect_and_preprocess_data
from model_selection_and_training import ModelTrainer, train_and_evaluate_model
from forecasting_and_prediction import TimeSeriesForecaster, generate_forecast

# Setup logging
logger = setup_logging()

def run_prophet_example():
    """Run a complete example of Prophet forecasting."""
    print("\n" + "="*80)
    print("FACEBOOK PROPHET FORECASTING EXAMPLE")
    print("="*80)
    
    # 1. Load configuration
    print("\nLoading configuration...")
    config = get_config()
    
    # Override model type to Prophet
    config.model.model_type = "prophet"
    
    # 2. Collect and preprocess data
    print("\nCollecting and preprocessing data...")
    symbol = "MSFT"  # Use Microsoft stock data
    
    try:
        data = collect_and_preprocess_data(config, symbol=symbol, save=True)
        print(f"Successfully collected data for {symbol}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"Error collecting data: {e}")
        data_path = Path(f'../data/preprocessed/{symbol}_preprocessed.csv')
        if data_path.exists():
            print(f"Loading cached data from {data_path}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            print("No data available. Please check your API key and connection.")
            return
    
    # 3. Create output directory
    output_dir = Path("../output/examples/prophet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Train a Prophet model
    print("\n" + "="*80)
    print("TRAINING PROPHET MODEL")
    print("="*80)
    
    # Create a model trainer
    trainer = ModelTrainer(config, data, target_column="close")
    
    # Split data
    train_data, test_data = trainer.split_data(test_size=0.2)
    print(f"Training data: {len(train_data)} records, Test data: {len(test_data)} records")
    
    # Train Prophet model with specific parameters
    print("\nTraining Prophet model...")
    trainer.train_prophet_model(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        exog_columns=["volume", "day_of_week", "month"]
    )
    
    # 5. Evaluate the model
    print("\nEvaluating model...")
    metrics = trainer.evaluate_model(exog_columns=["volume", "day_of_week", "month"])
    
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Plot actual vs predicted
    print("\nGenerating evaluation plot...")
    eval_plot_path = output_dir / "prophet_evaluation.png"
    trainer.plot_results(save_path=eval_plot_path)
    print(f"Evaluation plot saved to {eval_plot_path}")
    
    # 6. Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"prophet_{symbol}_{timestamp}.pkl"
    trainer.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # 7. Generate forecasts
    print("\n" + "="*80)
    print("GENERATING FORECASTS")
    print("="*80)
    
    # Create forecaster
    forecaster = TimeSeriesForecaster(config, model_path=model_path, data=data)
    
    # Generate forecast for 60 days
    forecast_days = 60
    print(f"\nGenerating {forecast_days}-day forecast...")
    forecast_df = forecaster.forecast(steps=forecast_days)
    
    print("\nForecast Summary (first 5 days):")
    print(forecast_df.head())
    
    # Plot the forecast
    print("\nGenerating forecast plot...")
    forecast_plot_path = output_dir / "prophet_forecast.png"
    forecaster.plot_forecast(forecast_df, historical_periods=90, save_path=forecast_plot_path)
    print(f"Forecast plot saved to {forecast_plot_path}")
    
    # 8. Analyze the forecast
    print("\nAnalyzing forecast...")
    analysis = forecaster.analyze_forecast(forecast_df)
    
    print("\nForecast Analysis:")
    for metric, value in analysis.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # 9. Identify potential disruptions
    print("\n" + "="*80)
    print("IDENTIFYING POTENTIAL DISRUPTIONS")
    print("="*80)
    
    print("\nDetecting potential disruptions...")
    disruptions_df = forecaster.detect_anomalies(forecast_df, threshold=2.0)
    
    # Count disruptions
    uncertain_periods = disruptions_df[disruptions_df["is_uncertain"]].sort_values(by="range_z_score", ascending=False)
    print(f"\nDetected {len(uncertain_periods)} periods with high uncertainty")
    
    if len(uncertain_periods) > 0:
        print("\nTop 5 most uncertain periods:")
        top_uncertain = uncertain_periods.head(5)
        for idx, row in top_uncertain.iterrows():
            print(f"  {idx.strftime('%Y-%m-%d')}: z-score = {row['range_z_score']:.2f}, " 
                  f"range = {row['forecast_range']:.2f}")
        
        # Plot disruptions
        print("\nGenerating disruption plot...")
        disruption_plot_path = output_dir / "prophet_disruptions.png"
        plt.figure(figsize=(14, 7))
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df["forecast"], label="Forecast", color="blue")
        
        # Mark disruptions
        plt.scatter(
            uncertain_periods.index, 
            uncertain_periods["forecast"],
            color="red", 
            s=50, 
            label="Potential Disruptions"
        )
        
        # Plot confidence intervals
        plt.fill_between(
            forecast_df.index,
            forecast_df["lower_bound"],
            forecast_df["upper_bound"],
            color="blue",
            alpha=0.2,
            label="95% Confidence Interval"
        )
        
        plt.title(f"Prophet Forecast with Potential Disruptions - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(disruption_plot_path)
        print(f"Disruption plot saved to {disruption_plot_path}")
    
    # 10. Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nProphet forecast for {symbol}:")
    print(f"  - Training data: {len(train_data)} days")
    print(f"  - Forecast horizon: {forecast_days} days")
    print(f"  - Model accuracy (RMSE): {metrics.get('rmse', 'N/A'):.4f}")
    print(f"  - Forecasted growth rate: {analysis.get('growth_rate_pct', 'N/A'):.2f}%")
    print(f"  - Potential disruptions: {len(uncertain_periods)}")
    
    print("\nAll outputs saved to:")
    print(f"  {output_dir}")
    
    print("\nTo generate forecasts with different parameters:")
    print("  python examples/prophet_example.py")
    
    print("\nThank you for using the Prophet forecasting example!")


if __name__ == "__main__":
    run_prophet_example() 