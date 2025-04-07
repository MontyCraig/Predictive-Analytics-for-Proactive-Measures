"""Module for identifying potential disruptions in time series data."""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from forecasting_and_prediction import TimeSeriesForecaster, generate_forecast
from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging

# Initialize logger
logger = setup_logging()


class DisruptionAnalyzer:
    """Analyzer for identifying potential disruptions in time series data."""

    def __init__(
        self,
        config: Config,
        forecast_data: Optional[pd.DataFrame] = None,
        historical_data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ):
        """Initialize the disruption analyzer.
        
        Args:
            config: Application configuration
            forecast_data: DataFrame with forecast data
            historical_data: DataFrame with historical data
            target_column: Target column to analyze
        """
        self.config = config
        self.forecast_data = forecast_data
        self.historical_data = historical_data
        self.target_column = target_column
        
        logger.info("DisruptionAnalyzer initialized")

    def load_forecast(self, file_path: Union[str, Path]) -> None:
        """Load forecast data from a CSV file.
        
        Args:
            file_path: Path to the forecast CSV file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Forecast file not found: {file_path}")
            raise FileNotFoundError(f"Forecast file not found: {file_path}")
            
        logger.info(f"Loading forecast data from {file_path}")
        
        try:
            self.forecast_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(self.forecast_data)} forecast records")
        except Exception as e:
            logger.error(f"Error loading forecast data: {str(e)}")
            raise

    def load_historical_data(self, file_path: Union[str, Path]) -> None:
        """Load historical data from a CSV file.
        
        Args:
            file_path: Path to the historical data CSV file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Historical data file not found: {file_path}")
            raise FileNotFoundError(f"Historical data file not found: {file_path}")
            
        logger.info(f"Loading historical data from {file_path}")
        
        try:
            self.historical_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(self.historical_data)} historical records")
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise

    def identify_trend_disruptions(
        self, window_size: int = 5, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Identify disruptions in the trend of the forecast.
        
        Args:
            window_size: Size of the rolling window for slope calculation
            threshold: Z-score threshold for disruption identification
            
        Returns:
            DataFrame with trend disruptions flagged
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise ValueError("Forecast data not loaded. Please load forecast data first.")
            
        logger.info(f"Identifying trend disruptions with threshold {threshold}")
        
        result_df = self.forecast_data.copy()
        
        # Calculate rolling slope
        x = np.arange(window_size)
        result_df["rolling_slope"] = np.nan
        
        for i in range(len(result_df) - window_size + 1):
            y = result_df["forecast"].iloc[i:i+window_size].values
            slope, _, _, _, _ = stats.linregress(x, y)
            result_df["rolling_slope"].iloc[i+window_size-1] = slope
            
        # Calculate slope z-scores
        valid_slopes = result_df["rolling_slope"].dropna()
        slope_mean = valid_slopes.mean()
        slope_std = valid_slopes.std()
        
        if slope_std > 0:  # Avoid division by zero
            result_df["slope_z_score"] = np.abs((result_df["rolling_slope"] - slope_mean) / slope_std)
            result_df["trend_disruption"] = result_df["slope_z_score"] > threshold
            
            n_disruptions = result_df["trend_disruption"].sum()
            logger.info(f"Detected {n_disruptions} trend disruptions")
        else:
            logger.warning("Slope standard deviation is zero. No trend disruptions detected.")
            result_df["slope_z_score"] = 0
            result_df["trend_disruption"] = False
            
        return result_df

    def identify_volatility_disruptions(
        self, window_size: int = 10, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Identify disruptions in the volatility of the forecast.
        
        Args:
            window_size: Size of the rolling window for volatility calculation
            threshold: Z-score threshold for disruption identification
            
        Returns:
            DataFrame with volatility disruptions flagged
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise ValueError("Forecast data not loaded. Please load forecast data first.")
            
        logger.info(f"Identifying volatility disruptions with threshold {threshold}")
        
        result_df = self.forecast_data.copy()
        
        # Calculate rolling volatility (standard deviation)
        result_df["rolling_volatility"] = result_df["forecast"].rolling(window=window_size).std()
        
        # Calculate volatility z-scores
        valid_volatility = result_df["rolling_volatility"].dropna()
        vol_mean = valid_volatility.mean()
        vol_std = valid_volatility.std()
        
        if vol_std > 0:  # Avoid division by zero
            result_df["volatility_z_score"] = np.abs((result_df["rolling_volatility"] - vol_mean) / vol_std)
            result_df["volatility_disruption"] = result_df["volatility_z_score"] > threshold
            
            n_disruptions = result_df["volatility_disruption"].sum()
            logger.info(f"Detected {n_disruptions} volatility disruptions")
        else:
            logger.warning("Volatility standard deviation is zero. No volatility disruptions detected.")
            result_df["volatility_z_score"] = 0
            result_df["volatility_disruption"] = False
            
        return result_df

    def identify_level_disruptions(
        self, threshold_std: float = 2.0
    ) -> pd.DataFrame:
        """Identify sudden level shifts in the forecast.
        
        Args:
            threshold_std: Number of standard deviations for level shift identification
            
        Returns:
            DataFrame with level shifts flagged
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise ValueError("Forecast data not loaded. Please load forecast data first.")
            
        logger.info(f"Identifying level disruptions with threshold {threshold_std} std")
        
        result_df = self.forecast_data.copy()
        
        # Calculate differences between consecutive forecasts
        result_df["forecast_diff"] = result_df["forecast"].diff()
        
        # Calculate statistics of differences
        diffs = result_df["forecast_diff"].dropna()
        diff_mean = diffs.mean()
        diff_std = diffs.std()
        
        # Identify level shifts
        threshold_value = diff_std * threshold_std
        result_df["level_disruption"] = np.abs(result_df["forecast_diff"] - diff_mean) > threshold_value
        
        n_disruptions = result_df["level_disruption"].sum()
        logger.info(f"Detected {n_disruptions} level disruptions")
            
        return result_df

    def identify_uncertainty_disruptions(
        self, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Identify periods with abnormal prediction uncertainty.
        
        Args:
            threshold: Z-score threshold for uncertainty identification
            
        Returns:
            DataFrame with uncertainty disruptions flagged
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise ValueError("Forecast data not loaded. Please load forecast data first.")
            
        if "lower_bound" not in self.forecast_data.columns or "upper_bound" not in self.forecast_data.columns:
            logger.error("Forecast data does not contain prediction intervals.")
            raise ValueError("Forecast data does not contain prediction intervals.")
            
        logger.info(f"Identifying uncertainty disruptions with threshold {threshold}")
        
        result_df = self.forecast_data.copy()
        
        # Calculate prediction interval width
        result_df["interval_width"] = result_df["upper_bound"] - result_df["lower_bound"]
        
        # Calculate z-scores of interval widths
        width_mean = result_df["interval_width"].mean()
        width_std = result_df["interval_width"].std()
        
        if width_std > 0:  # Avoid division by zero
            result_df["uncertainty_z_score"] = np.abs((result_df["interval_width"] - width_mean) / width_std)
            result_df["uncertainty_disruption"] = result_df["uncertainty_z_score"] > threshold
            
            n_disruptions = result_df["uncertainty_disruption"].sum()
            logger.info(f"Detected {n_disruptions} uncertainty disruptions")
        else:
            logger.warning("Interval width standard deviation is zero. No uncertainty disruptions detected.")
            result_df["uncertainty_z_score"] = 0
            result_df["uncertainty_disruption"] = False
            
        return result_df

    def identify_all_disruptions(self) -> pd.DataFrame:
        """Identify all types of disruptions in the forecast.
        
        Returns:
            DataFrame with all disruptions flagged
        """
        logger.info("Identifying all types of disruptions")
        
        # Identify trend disruptions
        trend_df = self.identify_trend_disruptions()
        
        # Identify volatility disruptions
        volatility_df = self.identify_volatility_disruptions()
        
        # Identify level disruptions
        level_df = self.identify_level_disruptions()
        
        # Identify uncertainty disruptions
        uncertainty_df = self.identify_uncertainty_disruptions()
        
        # Combine results
        result_df = self.forecast_data.copy()
        result_df["trend_disruption"] = trend_df["trend_disruption"]
        result_df["volatility_disruption"] = volatility_df["volatility_disruption"]
        result_df["level_disruption"] = level_df["level_disruption"]
        result_df["uncertainty_disruption"] = uncertainty_df["uncertainty_disruption"]
        
        # Count total disruptions
        result_df["total_disruptions"] = (
            result_df["trend_disruption"].astype(int) + 
            result_df["volatility_disruption"].astype(int) + 
            result_df["level_disruption"].astype(int) + 
            result_df["uncertainty_disruption"].astype(int)
        )
        
        # Flag significant disruptions (2 or more types)
        result_df["significant_disruption"] = result_df["total_disruptions"] >= 2
        
        n_significant = result_df["significant_disruption"].sum()
        logger.info(f"Detected {n_significant} significant disruptions")
            
        return result_df

    def plot_disruptions(
        self, 
        disruption_df: pd.DataFrame,
        historical_periods: int = 60,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot forecasts with disruptions highlighted.
        
        Args:
            disruption_df: DataFrame with disruptions flagged
            historical_periods: Number of historical periods to show
            save_path: Path to save the plot
        """
        if "significant_disruption" not in disruption_df.columns:
            logger.warning("Disruption DataFrame does not contain significant_disruption column.")
            return
            
        plt.figure(figsize=(14, 10))
        
        # Plot historical data if available
        if self.historical_data is not None:
            historical_data = self.historical_data.tail(historical_periods)
            plt.subplot(2, 1, 1)
            plt.plot(
                historical_data.index,
                historical_data[self.target_column],
                label="Historical Data",
                color="blue",
            )
            
        # Plot forecast
        plt.subplot(2, 1, 1)
        plt.plot(
            disruption_df.index,
            disruption_df["forecast"],
            label="Forecast",
            color="green",
            linestyle="-",
        )
        
        # Highlight significant disruptions
        significant_points = disruption_df[disruption_df["significant_disruption"]]
        if not significant_points.empty:
            plt.scatter(
                significant_points.index,
                significant_points["forecast"],
                color="red",
                s=100,
                label="Significant Disruptions",
                zorder=5,
            )
            
        # Add prediction intervals
        plt.fill_between(
            disruption_df.index,
            disruption_df["lower_bound"],
            disruption_df["upper_bound"],
            color="green",
            alpha=0.2,
            label="95% Confidence Interval",
        )
        
        plt.title("Forecast with Potential Disruptions")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Plot individual disruption types
        plt.subplot(2, 1, 2)
        
        width = 0.2
        x = np.arange(len(disruption_df))
        
        if "trend_disruption" in disruption_df.columns:
            plt.bar(
                x - 1.5*width, 
                disruption_df["trend_disruption"].astype(int), 
                width=width, 
                label="Trend Disruption"
            )
            
        if "volatility_disruption" in disruption_df.columns:
            plt.bar(
                x - 0.5*width, 
                disruption_df["volatility_disruption"].astype(int), 
                width=width, 
                label="Volatility Disruption"
            )
            
        if "level_disruption" in disruption_df.columns:
            plt.bar(
                x + 0.5*width, 
                disruption_df["level_disruption"].astype(int), 
                width=width, 
                label="Level Disruption"
            )
            
        if "uncertainty_disruption" in disruption_df.columns:
            plt.bar(
                x + 1.5*width, 
                disruption_df["uncertainty_disruption"].astype(int), 
                width=width, 
                label="Uncertainty Disruption"
            )
            
        plt.xticks(x, [d.strftime("%Y-%m-%d") for d in disruption_df.index], rotation=45)
        plt.title("Types of Disruptions by Date")
        plt.xlabel("Date")
        plt.ylabel("Disruption (0=No, 1=Yes)")
        plt.legend()
        plt.grid(True, axis="y")
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Disruption plot saved to {save_path}")
        else:
            plt.show()

    def generate_disruption_report(self, disruption_df: pd.DataFrame) -> Dict:
        """Generate a report summarizing the disruptions.
        
        Args:
            disruption_df: DataFrame with disruptions flagged
            
        Returns:
            Dictionary with disruption summary
        """
        logger.info("Generating disruption report")
        
        # Count disruptions by type
        trend_disruptions = disruption_df["trend_disruption"].sum() if "trend_disruption" in disruption_df.columns else 0
        volatility_disruptions = disruption_df["volatility_disruption"].sum() if "volatility_disruption" in disruption_df.columns else 0
        level_disruptions = disruption_df["level_disruption"].sum() if "level_disruption" in disruption_df.columns else 0
        uncertainty_disruptions = disruption_df["uncertainty_disruption"].sum() if "uncertainty_disruption" in disruption_df.columns else 0
        significant_disruptions = disruption_df["significant_disruption"].sum() if "significant_disruption" in disruption_df.columns else 0
        
        # Find dates of significant disruptions
        if "significant_disruption" in disruption_df.columns:
            significant_dates = disruption_df[disruption_df["significant_disruption"]].index.tolist()
        else:
            significant_dates = []
            
        # Calculate percentage of forecast periods with disruptions
        total_periods = len(disruption_df)
        disruption_percentage = (significant_disruptions / total_periods) * 100 if total_periods > 0 else 0
        
        # Create report
        report = {
            "trend_disruptions": int(trend_disruptions),
            "volatility_disruptions": int(volatility_disruptions),
            "level_disruptions": int(level_disruptions),
            "uncertainty_disruptions": int(uncertainty_disruptions),
            "significant_disruptions": int(significant_disruptions),
            "total_forecast_periods": total_periods,
            "disruption_percentage": disruption_percentage,
            "significant_disruption_dates": significant_dates,
        }
        
        return report


def analyze_disruptions(
    config: Config,
    forecast_path: Optional[Union[str, Path]] = None,
    historical_data_path: Optional[Union[str, Path]] = None,
    target_column: str = "close",
    generate_new_forecast: bool = False,
    forecast_steps: int = 30,
    plot: bool = True,
    save_results: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze potential disruptions in time series forecast.
    
    Args:
        config: Application configuration
        forecast_path: Path to forecast data file
        historical_data_path: Path to historical data file
        target_column: Target column to analyze
        generate_new_forecast: Whether to generate a new forecast
        forecast_steps: Number of steps to forecast if generating new forecast
        plot: Whether to plot the results
        save_results: Whether to save the results
        
    Returns:
        Tuple of (disruption_df, disruption_report)
    """
    # Initialize analyzer
    analyzer = DisruptionAnalyzer(config, target_column=target_column)
    
    # Load or generate forecast
    if generate_new_forecast or forecast_path is None:
        logger.info("Generating new forecast")
        forecast_df = generate_forecast(
            config,
            steps=forecast_steps,
            target_column=target_column,
            plot=False,
            save_forecast=True,
        )
        analyzer.forecast_data = forecast_df
    else:
        logger.info(f"Loading forecast from {forecast_path}")
        analyzer.load_forecast(forecast_path)
        
    # Load historical data if provided
    if historical_data_path:
        analyzer.load_historical_data(historical_data_path)
    elif historical_data_path is None and analyzer.historical_data is None:
        # Try to find latest preprocessed data
        preprocessed_dir = Path(config.output_dir) / "preprocessed"
        if preprocessed_dir.exists():
            data_files = list(preprocessed_dir.glob("*_preprocessed.csv"))
            if data_files:
                data_path = max(data_files, key=lambda p: p.stat().st_mtime)
                analyzer.load_historical_data(data_path)
        
    # Identify disruptions
    disruption_df = analyzer.identify_all_disruptions()
    
    # Generate report
    report = analyzer.generate_disruption_report(disruption_df)
    
    # Print report summary
    print("\nDisruption Analysis Summary:")
    print(f"Total forecast periods: {report['total_forecast_periods']}")
    print(f"Significant disruptions: {report['significant_disruptions']} ({report['disruption_percentage']:.2f}%)")
    print(f"Trend disruptions: {report['trend_disruptions']}")
    print(f"Volatility disruptions: {report['volatility_disruptions']}")
    print(f"Level disruptions: {report['level_disruptions']}")
    print(f"Uncertainty disruptions: {report['uncertainty_disruptions']}")
    
    if report['significant_disruption_dates']:
        print("\nSignificant disruption dates:")
        for date in report['significant_disruption_dates']:
            print(f"  - {date.strftime('%Y-%m-%d')}")
    
    # Plot disruptions
    if plot:
        analyzer.plot_disruptions(disruption_df)
    
    # Save results
    if save_results:
        # Save disruption data
        disruptions_dir = Path(config.output_dir) / "disruptions"
        disruptions_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disruption_path = disruptions_dir / f"disruptions_{timestamp}.csv"
        disruption_df.to_csv(disruption_path)
        logger.info(f"Disruption data saved to {disruption_path}")
        
        # Save plot
        if plot:
            plot_path = disruptions_dir / f"disruptions_plot_{timestamp}.png"
            analyzer.plot_disruptions(disruption_df, save_path=plot_path)
            
        # Save report as JSON
        import json
        from datetime import datetime
        
        # Convert datetime objects to strings for JSON serialization
        report_copy = report.copy()
        report_copy["significant_disruption_dates"] = [d.strftime("%Y-%m-%d") for d in report["significant_disruption_dates"]]
        
        report_path = disruptions_dir / f"disruption_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report_copy, f, indent=2)
        logger.info(f"Disruption report saved to {report_path}")
        
    return disruption_df, report


if __name__ == "__main__":
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Identify potential disruptions in time series forecasts")
    parser.add_argument("--forecast", help="Path to forecast data file")
    parser.add_argument("--historical", help="Path to historical data file")
    parser.add_argument("--target", default="close", help="Target column to analyze")
    parser.add_argument("--generate", action="store_true", help="Generate a new forecast")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to forecast")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    parser.add_argument("--no-plot", action="store_true", help="Do not display plots")
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.env_file)
    
    # Analyze disruptions
    disruption_df, report = analyze_disruptions(
        config,
        forecast_path=args.forecast,
        historical_data_path=args.historical,
        target_column=args.target,
        generate_new_forecast=args.generate,
        forecast_steps=args.steps,
        plot=not args.no_plot,
    )
