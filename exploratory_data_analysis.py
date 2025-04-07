"""Exploratory Data Analysis for time series data."""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging

# Use seaborn style for better visualizations
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

# Initialize logger
logger = setup_logging()


class TimeSeriesExplorer:
    """Explorer for time series data with visualization capabilities."""

    def __init__(self, data: pd.DataFrame, target_column: str = "close"):
        """Initialize the explorer with time series data.
        
        Args:
            data: Time series data
            target_column: Target column to analyze
        """
        self.data = data
        self.target_column = target_column
        
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not a DatetimeIndex. Converting...")
            self.data.index = pd.to_datetime(self.data.index)
            
        logger.info(f"Time Series Explorer initialized with {len(data)} data points")

    def plot_time_series(self, columns: Optional[List[str]] = None) -> None:
        """Plot the time series data.
        
        Args:
            columns: List of columns to plot (defaults to target_column)
        """
        if columns is None:
            columns = [self.target_column]
            
        plt.figure(figsize=(12, 6))
        
        for column in columns:
            if column in self.data.columns:
                plt.plot(self.data.index, self.data[column], label=column)
            else:
                logger.warning(f"Column {column} not found in data")
                
        plt.title(f"Time Series Plot for {', '.join(columns)}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_seasonal_decomposition(
        self, period: int = 30, model: str = "additive"
    ) -> None:
        """Plot seasonal decomposition of the time series.
        
        Args:
            period: Period for seasonal component
            model: Type of decomposition ('additive' or 'multiplicative')
        """
        logger.info(f"Performing {model} seasonal decomposition with period {period}")
        
        try:
            result = seasonal_decompose(
                self.data[self.target_column], model=model, period=period
            )
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            
            # Observed
            axes[0].plot(result.observed)
            axes[0].set_title("Observed")
            
            # Trend
            axes[1].plot(result.trend)
            axes[1].set_title("Trend")
            
            # Seasonal
            axes[2].plot(result.seasonal)
            axes[2].set_title("Seasonal")
            
            # Residual
            axes[3].plot(result.resid)
            axes[3].set_title("Residual")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {str(e)}")
            raise

    def plot_acf_pacf(self, lags: int = 40) -> None:
        """Plot autocorrelation and partial autocorrelation functions.
        
        Args:
            lags: Number of lags to include
        """
        logger.info(f"Plotting ACF and PACF with {lags} lags")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF
        plot_acf(self.data[self.target_column], lags=lags, ax=axes[0])
        axes[0].set_title(f"Autocorrelation Function (ACF) for {self.target_column}")
        
        # PACF
        plot_pacf(self.data[self.target_column], lags=lags, ax=axes[1])
        axes[1].set_title(f"Partial Autocorrelation Function (PACF) for {self.target_column}")
        
        plt.tight_layout()
        plt.show()

    def test_stationarity(self) -> Tuple[float, float, dict]:
        """Test stationarity of the time series using ADF test.
        
        Returns:
            Tuple of test statistic, p-value, and critical values
        """
        logger.info(f"Testing stationarity of {self.target_column}")
        
        result = adfuller(self.data[self.target_column].dropna())
        test_statistic, p_value, _, critical_values = result[:4]
        
        print(f"Augmented Dickey-Fuller Test for {self.target_column}")
        print(f"Test Statistic: {test_statistic:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"  {key}: {value:.4f}")
            
        if p_value < 0.05:
            print("\nResult: The time series is stationary (reject H0)")
        else:
            print("\nResult: The time series is non-stationary (fail to reject H0)")
            
        return test_statistic, p_value, critical_values

    def plot_rolling_statistics(self, window: int = 20) -> None:
        """Plot rolling mean and standard deviation.
        
        Args:
            window: Window size for rolling statistics
        """
        logger.info(f"Plotting rolling statistics with window {window}")
        
        rolling_mean = self.data[self.target_column].rolling(window=window).mean()
        rolling_std = self.data[self.target_column].rolling(window=window).std()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data[self.target_column], label=self.target_column)
        plt.plot(rolling_mean.index, rolling_mean, label=f"{window}-day Rolling Mean")
        plt.plot(rolling_std.index, rolling_std, label=f"{window}-day Rolling Std")
        plt.title(f"Rolling Statistics for {self.target_column}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_distribution(self) -> None:
        """Plot the distribution of the target column."""
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(self.data[self.target_column], kde=True)
        plt.title(f"Distribution of {self.target_column}")
        
        # QQ Plot
        plt.subplot(1, 2, 2)
        from scipy import stats
        stats.probplot(self.data[self.target_column], dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        
        plt.tight_layout()
        plt.show()

    def plot_box_plots(self, by: str = "month") -> None:
        """Plot box plots by time period.
        
        Args:
            by: Time period to group by ('month', 'quarter', 'year', 'day_of_week')
        """
        if by not in self.data.columns and by not in ["day_of_week", "month", "quarter", "year"]:
            logger.warning(f"Column {by} not found in data. Adding it...")
            
            if by == "day_of_week":
                self.data["day_of_week"] = self.data.index.dayofweek
            elif by == "month":
                self.data["month"] = self.data.index.month
            elif by == "quarter":
                self.data["quarter"] = self.data.index.quarter
            elif by == "year":
                self.data["year"] = self.data.index.year
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=by, y=self.target_column, data=self.data)
        plt.title(f"Box Plot of {self.target_column} by {by}")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, columns: Optional[List[str]] = None) -> None:
        """Plot correlation heatmap.
        
        Args:
            columns: List of columns to include (defaults to all numeric columns)
        """
        if columns is None:
            # Select only numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            columns = numeric_cols
            
        # Calculate correlation matrix
        corr_matrix = self.data[columns].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_lag_scatter(self, max_lag: int = 10) -> None:
        """Plot scatter plots between the series and its lags.
        
        Args:
            max_lag: Maximum lag to include
        """
        # Create lag columns if they don't exist
        for lag in range(1, max_lag + 1):
            lag_col = f"lag_{lag}"
            if lag_col not in self.data.columns:
                self.data[lag_col] = self.data[self.target_column].shift(lag)
                
        # Create scatter plots
        fig, axes = plt.subplots(
            nrows=(max_lag // 2) + (max_lag % 2), 
            ncols=2, 
            figsize=(14, max_lag * 2)
        )
        axes = axes.flatten()
        
        for lag in range(1, max_lag + 1):
            lag_col = f"lag_{lag}"
            ax = axes[lag - 1]
            ax.scatter(self.data[lag_col], self.data[self.target_column], alpha=0.5)
            ax.set_title(f"Lag {lag} vs {self.target_column}")
            ax.set_xlabel(f"Lag {lag}")
            ax.set_ylabel(self.target_column)
            
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self, output_dir: Optional[Path] = None) -> None:
        """Run a complete exploratory analysis with all plots.
        
        Args:
            output_dir: Directory to save plots (if None, plots are displayed)
        """
        logger.info("Starting full exploratory analysis")
        
        save_mode = output_dir is not None
        if save_mode:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving plots to {output_dir}")
            
        # Time series plot
        self.plot_time_series()
        if save_mode:
            plt.savefig(output_dir / "time_series_plot.png")
            plt.close()
            
        # Seasonal decomposition
        self.plot_seasonal_decomposition()
        if save_mode:
            plt.savefig(output_dir / "seasonal_decomposition.png")
            plt.close()
            
        # ACF and PACF
        self.plot_acf_pacf()
        if save_mode:
            plt.savefig(output_dir / "acf_pacf.png")
            plt.close()
            
        # Stationarity test
        self.test_stationarity()
        
        # Rolling statistics
        self.plot_rolling_statistics()
        if save_mode:
            plt.savefig(output_dir / "rolling_statistics.png")
            plt.close()
            
        # Distribution
        self.plot_distribution()
        if save_mode:
            plt.savefig(output_dir / "distribution.png")
            plt.close()
            
        # Box plots by month
        self.plot_box_plots()
        if save_mode:
            plt.savefig(output_dir / "box_plots_month.png")
            plt.close()
            
        # Correlation heatmap
        self.plot_heatmap()
        if save_mode:
            plt.savefig(output_dir / "correlation_heatmap.png")
            plt.close()
            
        # Lag scatter plots
        self.plot_lag_scatter()
        if save_mode:
            plt.savefig(output_dir / "lag_scatter.png")
            plt.close()
            
        logger.info("Full exploratory analysis completed")


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with time series data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform exploratory data analysis on time series data")
    parser.add_argument("--file", help="Path to data file")
    parser.add_argument("--symbol", default="MSFT", help="Stock symbol to analyze")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    parser.add_argument("--output-dir", help="Directory to save plots")
    args = parser.parse_args()
    
    # Load data
    if args.file:
        data = load_data(args.file)
    else:
        # Load config
        config = get_config(args.env_file)
        
        # Use preprocessed data from previous step
        from data_collection_and_preprocessing import collect_and_preprocess_data
        
        data = collect_and_preprocess_data(config, args.symbol)
        
    # Create explorer
    explorer = TimeSeriesExplorer(data)
    
    # Run full analysis
    output_dir = args.output_dir if args.output_dir else None
    explorer.run_full_analysis(output_dir)
