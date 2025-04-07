#!/usr/bin/env python3
"""
Example workflow for the Predictive Analytics for Proactive Measures framework.

This script demonstrates the complete workflow:
1. Data Collection and Preprocessing
2. Exploratory Data Analysis
3. Model Selection and Training
4. Forecasting and Prediction
5. Identifying Potential Disruptions
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')

# Load environment variables from the .env file
load_dotenv()

# Check if API key is available
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if not api_key:
    api_key = input("Enter your Alpha Vantage API key: ")
    os.environ['ALPHA_VANTAGE_API_KEY'] = api_key

# Import our modules
from utils.config import get_config
from data_collection_and_preprocessing import collect_and_preprocess_data
from exploratory_data_analysis import TimeSeriesExplorer
from model_selection_and_training import ModelTrainer
from forecasting_and_prediction import TimeSeriesForecaster
from identifying_potential_disruptions import DisruptionAnalyzer

def main():
    """Run the complete workflow."""
    # Load configuration
    config = get_config()
    
    print("=" * 80)
    print("PREDICTIVE ANALYTICS FOR PROACTIVE MEASURES")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model type: {config.model.model_type}")
    print(f"  Train/Test split: {config.model.train_size}/{config.model.test_size}")
    print(f"  Output directory: {config.output_dir}")
    print("")
    
    # Generate a timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Data Collection and Preprocessing
    print("=" * 80)
    print("STEP 1: DATA COLLECTION AND PREPROCESSING")
    print("=" * 80)
    
    # Choose a stock symbol
    symbol = 'MSFT'  # Microsoft
    
    # Collect and preprocess data
    try:
        print(f"Collecting and preprocessing data for {symbol}...")
        data = collect_and_preprocess_data(config, symbol=symbol, save=True)
        print(f"Successfully collected and preprocessed data for {symbol}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Features: {', '.join(data.columns)}")
    except Exception as e:
        print(f"Error collecting data: {e}")
        print("Trying to load from previously saved data...")
        
        # Try to load previously saved data
        try:
            data_path = Path(f'data/preprocessed/{symbol}_preprocessed.csv')
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            print(f"Successfully loaded data from {data_path}")
        except FileNotFoundError:
            print(f"No preprocessed data found for {symbol}. Please check your API key and connection.")
            return
    
    # Step 2: Exploratory Data Analysis
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    print("Creating TimeSeriesExplorer...")
    explorer = TimeSeriesExplorer(data, target_column='close')
    
    print("Testing for stationarity...")
    test_stat, p_value, critical_values = explorer.test_stationarity()
    print(f"ADF Test Statistic: {test_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")
    
    if p_value <= 0.05:
        print("\nThe time series is stationary (reject H0)")
    else:
        print("\nThe time series is non-stationary (fail to reject H0)")
    
    # Create output directory for plots
    plots_dir = Path('output/eda/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating time series plot...")
    explorer.plot_time_series()
    plt.savefig(plots_dir / f"{symbol}_time_series.png")
    
    print("Generating rolling statistics plot...")
    explorer.plot_rolling_statistics(window=20)
    plt.savefig(plots_dir / f"{symbol}_rolling_statistics.png")
    
    print("Generating seasonal decomposition plot...")
    explorer.plot_seasonal_decomposition(period=30, model='additive')
    plt.savefig(plots_dir / f"{symbol}_seasonal_decomposition.png")
    
    print("Generating ACF and PACF plots...")
    explorer.plot_acf_pacf(lags=40)
    plt.savefig(plots_dir / f"{symbol}_acf_pacf.png")
    
    print("Generating correlation heatmap...")
    explorer.plot_heatmap()
    plt.savefig(plots_dir / f"{symbol}_correlation_heatmap.png")
    
    print("Generating distribution plot...")
    explorer.plot_distribution()
    plt.savefig(plots_dir / f"{symbol}_distribution.png")
    
    print(f"EDA plots saved to {plots_dir}")
    
    # Step 3: Model Selection and Training
    print("\n" + "=" * 80)
    print("STEP 3: MODEL SELECTION AND TRAINING")
    print("=" * 80)
    
    print("Creating ModelTrainer...")
    trainer = ModelTrainer(config, data, target_column='close')
    
    print("Splitting data into training and test sets...")
    train_data, test_data = trainer.split_data()
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    print("\nTraining SARIMA model...")
    # Based on ACF/PACF plots, we specify appropriate orders
    trainer.train_sarima_model(
        order=(1, 1, 1),  # (p, d, q)
        seasonal_order=(1, 1, 1, 5),  # (P, D, Q, s)
        exog_columns=['lag_1', 'lag_5', 'rolling_mean_5']
    )
    print("Model training complete!")
    
    print("\nEvaluating model...")
    metrics = trainer.evaluate_model(exog_columns=['lag_1', 'lag_5', 'rolling_mean_5'])
    
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nGenerating results plot...")
    trainer.plot_results()
    plots_dir = Path('output/models/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{symbol}_model_evaluation.png")
    
    print("\nSaving model...")
    model_path = Path(f'models/sarima_{symbol}_{timestamp}.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Step 4: Forecasting and Prediction
    print("\n" + "=" * 80)
    print("STEP 4: FORECASTING AND PREDICTION")
    print("=" * 80)
    
    print("Creating TimeSeriesForecaster...")
    forecaster = TimeSeriesForecaster(config, model_path=model_path, data=data, target_column='close')
    
    print("Generating forecast for the next 30 days...")
    forecast_df = forecaster.forecast(steps=30, confidence_interval=0.95)
    print("Forecast generated successfully!")
    
    print("\nForecast Summary:")
    print(forecast_df.head())
    
    print("\nAnalyzing forecast...")
    analysis = forecaster.analyze_forecast(forecast_df)
    
    print("Forecast Analysis:")
    for metric, value in analysis.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nGenerating forecast plot...")
    forecaster.plot_forecast(forecast_df, historical_periods=60)
    plots_dir = Path('output/forecasts/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{symbol}_forecast_30days.png")
    
    print("\nSaving forecast...")
    forecast_path = Path(f'output/forecasts/data/forecast_{symbol}_30days_{timestamp}.csv')
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_path)
    print(f"Forecast saved to {forecast_path}")
    
    # Step 5: Identifying Potential Disruptions
    print("\n" + "=" * 80)
    print("STEP 5: IDENTIFYING POTENTIAL DISRUPTIONS")
    print("=" * 80)
    
    print("Creating DisruptionAnalyzer...")
    analyzer = DisruptionAnalyzer(config, forecast_data=forecast_df, historical_data=data, target_column='close')
    
    print("Identifying trend disruptions...")
    trend_df = analyzer.identify_trend_disruptions(window_size=5, threshold=2.0)
    print(f"Identified {trend_df['trend_disruption'].sum()} trend disruptions")
    
    print("Identifying volatility disruptions...")
    volatility_df = analyzer.identify_volatility_disruptions(window_size=10, threshold=2.0)
    print(f"Identified {volatility_df['volatility_disruption'].sum()} volatility disruptions")
    
    print("Identifying level disruptions...")
    level_df = analyzer.identify_level_disruptions(threshold_std=2.0)
    print(f"Identified {level_df['level_disruption'].sum()} level disruptions")
    
    print("Identifying uncertainty disruptions...")
    try:
        uncertainty_df = analyzer.identify_uncertainty_disruptions(threshold=2.0)
        print(f"Identified {uncertainty_df['uncertainty_disruption'].sum()} uncertainty disruptions")
    except Exception as e:
        print(f"Could not identify uncertainty disruptions: {e}")
    
    print("\nIdentifying all disruptions...")
    disruption_df = analyzer.identify_all_disruptions()
    print("All disruptions identified successfully!")
    
    print("\nGenerating disruption report...")
    report = analyzer.generate_disruption_report(disruption_df)
    
    print("Disruption Analysis Report:")
    print(f"Total forecast periods: {report['total_forecast_periods']}")
    print(f"Periods with any disruption: {report['periods_with_disruptions']}")
    print(f"Significant disruptions: {report['significant_disruptions']}")
    print(f"Disruption percentage: {report['disruption_percentage']:.2f}%")
    
    print("\nDistribution of disruption types:")
    for disruption_type, count in report['disruption_type_counts'].items():
        print(f"  {disruption_type}: {count}")
    
    if report.get('significant_disruption_dates'):
        print("\nDates with significant disruptions:")
        for date in report['significant_disruption_dates']:
            print(f"  {date.strftime('%Y-%m-%d')}")
    
    print("\nGenerating disruption plot...")
    analyzer.plot_disruptions(disruption_df, historical_periods=60)
    plots_dir = Path('output/disruptions/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{symbol}_disruptions.png")
    
    print("\nSaving disruption analysis...")
    disruption_path = Path(f'output/disruptions/data/disruption_analysis_{symbol}_{timestamp}.csv')
    disruption_path.parent.mkdir(parents=True, exist_ok=True)
    disruption_df.to_csv(disruption_path)
    print(f"Disruption analysis saved to {disruption_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    
    print(f"Analysis completed for {symbol}")
    print(f"Timestamp: {timestamp}")
    print("\nOutputs:")
    print(f"  - Model: {model_path}")
    print(f"  - Forecast: {forecast_path}")
    print(f"  - Disruption Analysis: {disruption_path}")
    print(f"  - Plots: output/*/plots/")
    
    print("\nSummary of findings:")
    print(f"  - Model performance (RMSE): {metrics.get('rmse', 'N/A')}")
    print(f"  - Forecast growth rate: {analysis.get('growth_rate_pct', 'N/A')}%")
    print(f"  - Forecast volatility: {analysis.get('volatility', 'N/A')}")
    print(f"  - Disruption percentage: {report['disruption_percentage']:.2f}%")
    
    print("\nThank you for using Predictive Analytics for Proactive Measures!")

if __name__ == "__main__":
    main() 