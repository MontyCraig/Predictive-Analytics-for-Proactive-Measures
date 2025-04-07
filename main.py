#!/usr/bin/env python3
"""Main entry point for Predictive Analytics for Proactive Measures.

This script ties together all components of the workflow:
1. Data collection and preprocessing
2. Exploratory data analysis
3. Model selection and training
4. Forecasting and prediction
5. Identifying potential disruptions
"""
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from data_collection_and_preprocessing import collect_and_preprocess_data
from exploratory_data_analysis import TimeSeriesExplorer
from forecasting_and_prediction import generate_forecast
from identifying_potential_disruptions import analyze_disruptions
from model_selection_and_training import train_and_evaluate_model
from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging

# Initialize logger
logger = setup_logging()


def setup_environment() -> None:
    """Set up the environment by creating necessary directories."""
    logger.info("Setting up environment")
    
    # Define all required directories
    required_dirs = [
        "data",
        "models",
        "output",
        "output/raw",
        "output/preprocessed",
        "output/plots",
        "output/models",
        "output/forecasts/data",
        "output/forecasts/plots",
        "output/disruptions",
    ]
    
    # Create directories
    for directory in required_dirs:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
    logger.info("Environment setup complete")


def create_env_file(api_key: str = None) -> Path:
    """Create a .env file with API keys.
    
    Args:
        api_key: Alpha Vantage API key
        
    Returns:
        Path to the created .env file
    """
    if api_key is None:
        # Get API key from user
        api_key = input("Enter your Alpha Vantage API key (or leave blank for demo): ")
        if not api_key:
            api_key = "demo"
            
    # Create .env file
    env_path = Path(".env")
    with open(env_path, "w") as f:
        f.write(f"ALPHA_VANTAGE_API_KEY={api_key}\n")
        f.write("MODEL_TYPE=sarima\n")
        f.write("TRAIN_SIZE=0.8\n")
        f.write("TEST_SIZE=0.2\n")
        f.write("RANDOM_STATE=42\n")
        f.write("LOG_LEVEL=INFO\n")
        
    logger.info(f"Created .env file at {env_path.absolute()}")
    return env_path


def run_full_workflow(
    config: Config,
    symbol: str = "MSFT",
    output_dir: Optional[str] = None,
    skip_collection: bool = False,
    skip_eda: bool = False,
    skip_training: bool = False,
    skip_forecasting: bool = False,
    skip_disruptions: bool = False,
    show_plots: bool = False,
    forecast_steps: int = 30,
) -> None:
    """Run the full predictive analytics workflow.
    
    Args:
        config: Configuration object
        symbol: Stock symbol to analyze
        output_dir: Directory to save outputs
        skip_collection: Whether to skip data collection
        skip_eda: Whether to skip exploratory data analysis
        skip_training: Whether to skip model training
        skip_forecasting: Whether to skip forecasting
        skip_disruptions: Whether to skip disruption analysis
        show_plots: Whether to show plots
        forecast_steps: Number of steps to forecast
    """
    # Update output directory if provided
    if output_dir:
        config.output_dir = output_dir
        
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting full workflow for symbol {symbol}")
    logger.info(f"Outputs will be saved to {run_dir}")
    
    # Step 1: Data Collection and Preprocessing
    if not skip_collection:
        logger.info("Step 1: Data Collection and Preprocessing")
        data = collect_and_preprocess_data(config, symbol, save=True)
        data_path = Path(config.output_dir) / "preprocessed" / f"{symbol}_preprocessed.csv"
    else:
        logger.info("Skipping data collection")
        # Try to find existing data
        data_path = Path(config.output_dir) / "preprocessed" / f"{symbol}_preprocessed.csv"
        if data_path.exists():
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            logger.error(f"No preprocessed data found at {data_path}")
            raise FileNotFoundError(f"No preprocessed data found at {data_path}")
    
    # Step 2: Exploratory Data Analysis
    if not skip_eda:
        logger.info("Step 2: Exploratory Data Analysis")
        explorer = TimeSeriesExplorer(data)
        eda_dir = run_dir / "eda"
        eda_dir.mkdir(exist_ok=True)
        explorer.run_full_analysis(eda_dir if not show_plots else None)
    else:
        logger.info("Skipping exploratory data analysis")
    
    # Step 3: Model Selection and Training
    if not skip_training:
        logger.info("Step 3: Model Selection and Training")
        metrics = train_and_evaluate_model(
            config,
            data=data,
            target_column="close",
            exog_columns=["lag_1", "lag_5", "rolling_mean_5"],
            save_model=True,
            save_plot=not show_plots,
        )
        
        # Save metrics to file
        metrics_path = run_dir / "model_metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Model Evaluation Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
    else:
        logger.info("Skipping model training")
    
    # Step 4: Forecasting and Prediction
    if not skip_forecasting:
        logger.info("Step 4: Forecasting and Prediction")
        forecast_df = generate_forecast(
            config,
            steps=forecast_steps,
            target_column="close",
            plot=True,
            save_forecast=True,
            save_plot=not show_plots,
        )
    else:
        logger.info("Skipping forecasting")
        forecast_df = None
    
    # Step 5: Identifying Potential Disruptions
    if not skip_disruptions:
        logger.info("Step 5: Identifying Potential Disruptions")
        disruption_df, report = analyze_disruptions(
            config,
            target_column="close",
            generate_new_forecast=forecast_df is None,
            forecast_steps=forecast_steps,
            plot=True,
            save_results=True,
        )
        
        # Save report to run directory
        report_path = run_dir / "disruption_report.txt"
        with open(report_path, "w") as f:
            f.write("Disruption Analysis Report:\n")
            f.write(f"Total forecast periods: {report['total_forecast_periods']}\n")
            f.write(f"Significant disruptions: {report['significant_disruptions']} ({report['disruption_percentage']:.2f}%)\n")
            f.write(f"Trend disruptions: {report['trend_disruptions']}\n")
            f.write(f"Volatility disruptions: {report['volatility_disruptions']}\n")
            f.write(f"Level disruptions: {report['level_disruptions']}\n")
            f.write(f"Uncertainty disruptions: {report['uncertainty_disruptions']}\n")
            
            if report['significant_disruption_dates']:
                f.write("\nSignificant disruption dates:\n")
                for date in report['significant_disruption_dates']:
                    f.write(f"  - {date.strftime('%Y-%m-%d')}\n")
    else:
        logger.info("Skipping disruption analysis")
    
    logger.info("Workflow complete")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predictive Analytics for Proactive Measures")
    parser.add_argument("--symbol", default="MSFT", help="Stock symbol to analyze")
    parser.add_argument("--api-key", help="Alpha Vantage API key")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-eda", action="store_true", help="Skip exploratory data analysis")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-forecasting", action="store_true", help="Skip forecasting")
    parser.add_argument("--skip-disruptions", action="store_true", help="Skip disruption analysis")
    parser.add_argument("--show-plots", action="store_true", help="Show plots instead of saving")
    parser.add_argument("--forecast-steps", type=int, default=30, help="Number of steps to forecast")
    args = parser.parse_args()
    
    # Set up environment
    setup_environment()
    
    # Create .env file if needed
    if not args.env_file and not os.path.exists(".env"):
        env_path = create_env_file(args.api_key)
        args.env_file = str(env_path)
    
    # Load config
    config = get_config(args.env_file)
    
    # Run workflow
    run_full_workflow(
        config,
        symbol=args.symbol,
        output_dir=args.output_dir,
        skip_collection=args.skip_collection,
        skip_eda=args.skip_eda,
        skip_training=args.skip_training,
        skip_forecasting=args.skip_forecasting,
        skip_disruptions=args.skip_disruptions,
        show_plots=args.show_plots,
        forecast_steps=args.forecast_steps,
    ) 