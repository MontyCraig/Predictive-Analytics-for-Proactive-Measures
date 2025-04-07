"""Data collection and preprocessing module for Predictive Analytics."""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from utils.config import Config, get_config
from utils.logging_config import logger, setup_logging

# Initialize logger
logger = setup_logging()


class TimeSeriesData(BaseModel):
    """Pydantic model for time series data."""

    symbol: str = Field(..., description="Stock symbol or identifier")
    interval: str = Field(..., description="Data interval (daily, weekly, monthly)")
    time_series: Dict[str, Dict[str, str]] = Field(
        ..., description="Time series data points"
    )
    last_refreshed: datetime = Field(..., description="Last data refresh timestamp")
    output_size: str = Field(..., description="Output size (compact or full)")
    time_zone: str = Field(..., description="Time zone of the data")


class AlphaVantageClient:
    """Client for Alpha Vantage API."""

    def __init__(self, config: Config):
        """Initialize the Alpha Vantage client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.base_url = config.api.alpha_vantage_base_url
        self.api_key = config.api.alpha_vantage_api_key.get_secret_value()
        self.session = requests.Session()
        
        logger.info("Alpha Vantage client initialized")

    def get_time_series(
        self,
        symbol: str,
        function: str = "TIME_SERIES_DAILY",
        interval: Optional[str] = None,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """Fetch time series data from Alpha Vantage.
        
        Args:
            symbol: Stock ticker symbol
            function: API function to call
            interval: Data interval for intraday data
            outputsize: Output size (compact or full)
            
        Returns:
            DataFrame with time series data
            
        Raises:
            RequestException: If there's an error with the API request
            ValueError: If the response format is unexpected
        """
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
        }
        
        if interval and function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval
            
        try:
            logger.info(f"Fetching {function} data for {symbol}")
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle API limit
            if "Note" in data:
                logger.warning("API call frequency limit reached. Waiting for 60 seconds.")
                time.sleep(60)
                return self.get_time_series(symbol, function, interval, outputsize)
                
            # Extract time series data based on the function
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({interval})"
            elif function == "TIME_SERIES_DAILY":
                time_series_key = "Time Series (Daily)"
            elif function == "TIME_SERIES_WEEKLY":
                time_series_key = "Weekly Time Series"
            elif function == "TIME_SERIES_MONTHLY":
                time_series_key = "Monthly Time Series"
            else:
                raise ValueError(f"Unsupported function: {function}")
                
            if time_series_key not in data:
                logger.error(f"Unexpected API response format: {data}")
                raise ValueError(f"Time series key '{time_series_key}' not found in response")
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            
            # Rename columns to remove prefixes
            df.columns = [col.split(". ")[1] if ". " in col else col for col in df.columns]
            
            # Convert data types
            for col in df.columns:
                if col != "volume":
                    df[col] = pd.to_numeric(df[col])
                else:
                    df["volume"] = pd.to_numeric(df["volume"])
                    
            # Set index as datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Add symbol and function metadata
            df["symbol"] = symbol
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Alpha Vantage: {str(e)}")
            raise
        except (ValueError, KeyError) as e:
            logger.error(f"Error processing Alpha Vantage response: {str(e)}")
            raise ValueError(f"Failed to process Alpha Vantage response: {str(e)}")

    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """Save data to a CSV file.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the data
            
        Raises:
            IOError: If there's an error saving the file
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(file_path)
            logger.info(f"Data saved to {file_path}")
        except IOError as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the time series data.
    
    Args:
        df: Raw time series data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values with linear interpolation
    if data.isna().any().any():
        logger.info("Handling missing values with linear interpolation")
        data = data.interpolate(method="linear")
        
    # Handle remaining NaN values (e.g., at the beginning of the series)
    data = data.fillna(method="bfill")
    
    # Add date-based features
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["day_of_week"] = data.index.dayofweek
    data["quarter"] = data.index.quarter
    
    # Add lag features
    logger.info("Adding lag features")
    data["lag_1"] = data["close"].shift(1)
    data["lag_5"] = data["close"].shift(5)
    data["lag_10"] = data["close"].shift(10)
    
    # Add rolling statistics
    logger.info("Adding rolling statistics")
    data["rolling_mean_5"] = data["close"].rolling(window=5).mean()
    data["rolling_mean_10"] = data["close"].rolling(window=10).mean()
    data["rolling_std_5"] = data["close"].rolling(window=5).std()
    data["rolling_std_10"] = data["close"].rolling(window=10).std()
    
    # Calculate returns
    logger.info("Calculating returns")
    data["daily_return"] = data["close"].pct_change()
    data["log_return"] = data["daily_return"].apply(lambda x: 0 if x <= -1 else np.log(x + 1))
    
    # Drop rows with NaN values from lag and rolling features
    data = data.dropna()
    
    logger.info(f"Preprocessing complete. Data shape: {data.shape}")
    return data


def collect_and_preprocess_data(
    config: Config, symbol: str = "MSFT", save: bool = True
) -> pd.DataFrame:
    """Collect and preprocess data.
    
    Args:
        config: Application configuration
        symbol: Stock symbol to fetch data for
        save: Whether to save the data to a file
        
    Returns:
        Preprocessed DataFrame
    """
    client = AlphaVantageClient(config)
    
    # Fetch data
    df = client.get_time_series(symbol=symbol)
    
    if save:
        raw_file_path = Path(config.output_dir) / "raw" / f"{symbol}_raw.csv"
        client.save_data(df, raw_file_path)
    
    # Preprocess data
    preprocessed_df = preprocess_data(df)
    
    if save:
        preprocessed_file_path = Path(config.output_dir) / "preprocessed" / f"{symbol}_preprocessed.csv"
        client.save_data(preprocessed_df, preprocessed_file_path)
    
    return preprocessed_df


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Collect and preprocess time series data")
    parser.add_argument("--symbol", default="MSFT", help="Stock symbol to fetch data for")
    parser.add_argument("--env-file", help="Path to .env file with API keys")
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.env_file)
    
    # Collect and preprocess data
    data = collect_and_preprocess_data(config, args.symbol)
    
    # Print summary
    print("\nData Summary:")
    print(f"Symbol: {args.symbol}")
    print(f"Date Range: {data.index.min()} to {data.index.max()}")
    print(f"Number of Records: {len(data)}")
    print(f"Columns: {', '.join(data.columns)}")
    print("\nSample Data:")
    print(data.head())
