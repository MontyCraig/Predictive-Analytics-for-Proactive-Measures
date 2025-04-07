"""Helper utilities for Predictive Analytics for Proactive Measures.

This module provides common utility functions for string formatting, datetime operations,
file handling, and other miscellaneous helper functions used throughout the project.
"""

import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from utils.logging_config import logger


# ==============================
# String Formatting Functions
# ==============================

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case string to camelCase.
    
    Args:
        snake_str: String in snake_case format
        
    Returns:
        String in camelCase format
    """
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def snake_to_title(snake_str: str) -> str:
    """Convert snake_case string to Title Case.
    
    Args:
        snake_str: String in snake_case format
        
    Returns:
        String in Title Case format
    """
    return ' '.join(word.capitalize() for word in snake_str.split('_'))


def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """Truncate a string to a maximum length and add a suffix if truncated.
    
    Args:
        text: String to truncate
        max_length: Maximum length of the string (default: 50)
        suffix: Suffix to add if truncated (default: '...')
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_number(number: float, precision: int = 2, use_thousands_separator: bool = True) -> str:
    """Format a number with specified precision and optional thousands separator.
    
    Args:
        number: Number to format
        precision: Number of decimal places (default: 2)
        use_thousands_separator: Whether to use thousands separator (default: True)
        
    Returns:
        Formatted number as string
    """
    if use_thousands_separator:
        return f"{number:,.{precision}f}"
    return f"{number:.{precision}f}"


def format_percent(value: float, precision: int = 2) -> str:
    """Format a value as a percentage string.
    
    Args:
        value: Value to format as percentage (0.1 = 10%)
        precision: Number of decimal places (default: 2)
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


# ==============================
# DateTime Helper Functions
# ==============================

def parse_date_string(date_str: str, formats: Optional[List[str]] = None) -> Optional[datetime]:
    """Parse a date string using multiple possible formats.
    
    Args:
        date_str: Date string to parse
        formats: List of formats to try (default: common formats)
        
    Returns:
        Parsed datetime object or None if parsing failed
    """
    if formats is None:
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_str}")
    return None


def get_date_range(start_date: Union[str, datetime], 
                  end_date: Union[str, datetime], 
                  freq: str = 'D') -> pd.DatetimeIndex:
    """Get a range of dates between start and end dates.
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        freq: Frequency of dates (default: 'D' for days)
        
    Returns:
        DatetimeIndex with the date range
    """
    if isinstance(start_date, str):
        start_date = parse_date_string(start_date)
    
    if isinstance(end_date, str):
        end_date = parse_date_string(end_date)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def add_business_days(date: datetime, n_days: int) -> datetime:
    """Add business days to a date, skipping weekends.
    
    Args:
        date: Starting date
        n_days: Number of business days to add
        
    Returns:
        Datetime with business days added
    """
    if n_days < 0:
        return subtract_business_days(date, abs(n_days))
    
    result = date
    day_count = 0
    
    while day_count < n_days:
        result += timedelta(days=1)
        if result.weekday() < 5:  # Monday to Friday are 0 to 4
            day_count += 1
            
    return result


def subtract_business_days(date: datetime, n_days: int) -> datetime:
    """Subtract business days from a date, skipping weekends.
    
    Args:
        date: Starting date
        n_days: Number of business days to subtract
        
    Returns:
        Datetime with business days subtracted
    """
    if n_days < 0:
        return add_business_days(date, abs(n_days))
    
    result = date
    day_count = 0
    
    while day_count < n_days:
        result -= timedelta(days=1)
        if result.weekday() < 5:  # Monday to Friday are 0 to 4
            day_count += 1
            
    return result


def is_business_day(date: datetime) -> bool:
    """Check if a date is a business day (Monday through Friday).
    
    Args:
        date: Date to check
        
    Returns:
        True if the date is a business day, False otherwise
    """
    return date.weekday() < 5  # Monday to Friday are 0 to 4


def current_quarter(date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Get the start and end dates of the current quarter for a given date.
    
    Args:
        date: Date to get quarter for (default: current date)
        
    Returns:
        Tuple of (quarter_start, quarter_end) dates
    """
    if date is None:
        date = datetime.now()
    
    quarter = (date.month - 1) // 3 + 1
    start_month = (quarter - 1) * 3 + 1
    
    start_date = datetime(date.year, start_month, 1)
    
    if quarter < 4:
        end_date = datetime(date.year, start_month + 3, 1) - timedelta(days=1)
    else:
        end_date = datetime(date.year + 1, 1, 1) - timedelta(days=1)
    
    return start_date, end_date


# ==============================
# File and Path Helpers
# ==============================

def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_latest_file(directory: Union[str, Path], pattern: str = '*') -> Optional[Path]:
    """Get the most recently modified file in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match files (default: '*')
        
    Returns:
        Path to the latest file or None if no files match
    """
    directory = Path(directory)
    matching_files = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    return matching_files[0] if matching_files else None


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """Calculate the hash of a file using the specified algorithm.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: 'md5')
        
    Returns:
        Hash digest as a hexadecimal string
    """
    file_path = Path(file_path)
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def save_json(data: Any, file_path: Union[str, Path], pretty: bool = True) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the file
        pretty: Whether to format the JSON for readability (default: True)
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with open(file_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=True)
        else:
            json.dump(data, f)


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        return json.load(f)


# ==============================
# Data Processing Helpers
# ==============================

def detect_outliers(series: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """Detect outliers in a series using various methods.
    
    Args:
        series: Series to detect outliers in
        method: Method to use ('zscore', 'iqr', or 'percentile')
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean series with True for outliers
    """
    if method == 'zscore':
        z_scores = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold
    
    elif method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'percentile':
        lower_bound = series.quantile(threshold / 100)
        upper_bound = series.quantile(1 - threshold / 100)
        return (series < lower_bound) | (series > upper_bound)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def infer_frequency(time_index: pd.DatetimeIndex) -> str:
    """Infer the frequency of a time series from its index.
    
    Args:
        time_index: DatetimeIndex of the time series
        
    Returns:
        Inferred frequency as a string
    """
    if len(time_index) < 3:
        return 'unknown'
    
    # Calculate the differences between consecutive dates
    diff = pd.Series(time_index[1:]) - pd.Series(time_index[:-1])
    
    # Get the most common difference
    most_common_diff = diff.mode()[0]
    days = most_common_diff.days
    
    if days == 1:
        return 'D'  # Daily
    elif days == 7:
        return 'W'  # Weekly
    elif 28 <= days <= 31:
        return 'M'  # Monthly
    elif 90 <= days <= 92:
        return 'Q'  # Quarterly
    elif 365 <= days <= 366:
        return 'Y'  # Yearly
    else:
        hours = most_common_diff.seconds // 3600
        if hours == 1:
            return 'H'  # Hourly
        elif hours == 24:
            return 'D'  # Daily
        
    return 'unknown'


def moving_average(series: pd.Series, window: int, center: bool = False) -> pd.Series:
    """Calculate the moving average of a series.
    
    Args:
        series: Series to calculate moving average for
        window: Window size
        center: Whether to center the window (default: False)
        
    Returns:
        Series with moving average
    """
    return series.rolling(window=window, center=center).mean()


def exponential_smoothing(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Apply exponential smoothing to a series.
    
    Args:
        series: Series to smooth
        alpha: Smoothing factor (default: 0.3)
        
    Returns:
        Smoothed series
    """
    return series.ewm(alpha=alpha).mean()


# ==============================
# Miscellaneous Helpers
# ==============================

def generate_id(prefix: str = '', length: int = 8) -> str:
    """Generate a random ID with an optional prefix.
    
    Args:
        prefix: Prefix for the ID (default: '')
        length: Length of the random part (default: 8)
        
    Returns:
        Generated ID
    """
    import random
    import string
    
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    random_part = ''.join(random.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def retry(func, max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: Tuple = (Exception,)):
    """Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Backoff multiplier (default: 2.0)
        exceptions: Exceptions to catch (default: Exception)
        
    Returns:
        Result of the function
    """
    import time
    
    attempt = 0
    while attempt < max_attempts:
        try:
            return func()
        except exceptions as e:
            attempt += 1
            if attempt == max_attempts:
                raise
            
            sleep_time = delay * (backoff ** (attempt - 1))
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)


def memoize(func):
    """Memoize a function to cache its results.
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    def memoized(*args, **kwargs):
        # Create a key from the function arguments
        key_args = tuple(args)
        key_kwargs = tuple(sorted(kwargs.items()))
        key = (key_args, key_kwargs)
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    return memoized 