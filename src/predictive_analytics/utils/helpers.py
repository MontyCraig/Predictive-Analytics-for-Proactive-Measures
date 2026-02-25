"""General-purpose helper utilities for the Predictive Analytics package.

This module provides common utility functions organised into four categories:

* **String formatting** -- case conversion, truncation, number formatting.
* **DateTime operations** -- flexible parsing, date ranges, business-day
  arithmetic, quarter boundaries.
* **File and path helpers** -- directory creation, latest-file discovery,
  hashing, JSON I/O.
* **Data processing** -- outlier detection, frequency inference, moving
  average, exponential smoothing.
* **Miscellaneous** -- ID generation, retry with back-off, memoisation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import string
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from predictive_analytics.config.logging import setup_logging

__all__: list[str] = [
    "snake_to_camel",
    "snake_to_title",
    "truncate_string",
    "format_number",
    "format_percent",
    "parse_date_string",
    "get_date_range",
    "add_business_days",
    "subtract_business_days",
    "is_business_day",
    "current_quarter",
    "ensure_directory",
    "get_latest_file",
    "get_file_hash",
    "save_json",
    "load_json",
    "detect_outliers",
    "infer_frequency",
    "moving_average",
    "exponential_smoothing",
    "generate_id",
    "retry",
    "memoize",
]

logger: logging.Logger = setup_logging()

_T = TypeVar("_T")
_F = TypeVar("_F", bound=Callable[..., Any])


# ======================================================================
# String Formatting Functions
# ======================================================================


def snake_to_camel(snake_str: str) -> str:
    """Convert a *snake_case* string to *camelCase*.

    Args:
        snake_str: String in ``snake_case`` format.

    Returns:
        String in ``camelCase`` format.
    """
    components: list[str] = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def snake_to_title(snake_str: str) -> str:
    """Convert a *snake_case* string to *Title Case*.

    Args:
        snake_str: String in ``snake_case`` format.

    Returns:
        String in ``Title Case`` format.
    """
    return " ".join(word.capitalize() for word in snake_str.split("_"))


def truncate_string(
    text: str, max_length: int = 50, suffix: str = "..."
) -> str:
    """Truncate *text* to at most *max_length* characters.

    If truncation occurs the *suffix* is appended (and its length is
    accounted for so the total never exceeds *max_length*).

    Args:
        text: The string to truncate.
        max_length: Maximum allowed length.
        suffix: Suffix appended when the string is truncated.

    Returns:
        The (possibly truncated) string.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_number(
    number: float,
    precision: int = 2,
    use_thousands_separator: bool = True,
) -> str:
    """Format a number with decimal precision and optional grouping.

    Args:
        number: The value to format.
        precision: Number of decimal places.
        use_thousands_separator: Insert comma grouping separators.

    Returns:
        The formatted number as a string.
    """
    if use_thousands_separator:
        return f"{number:,.{precision}f}"
    return f"{number:.{precision}f}"


def format_percent(value: float, precision: int = 2) -> str:
    """Format a decimal fraction as a percentage string.

    ``0.1`` becomes ``'10.00%'`` (with the default precision).

    Args:
        value: Fractional value (e.g. ``0.1`` for 10 %).
        precision: Decimal places in the output.

    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{precision}f}%"


# ======================================================================
# DateTime Helper Functions
# ======================================================================


def parse_date_string(
    date_str: str,
    formats: Optional[List[str]] = None,
) -> Optional[datetime]:
    """Parse a date string by trying several formats.

    Args:
        date_str: The date string to parse.
        formats: Candidate ``strptime`` format strings.  Defaults to a
            built-in set of common formats.

    Returns:
        The parsed ``datetime``, or *None* if none of the formats match.
    """
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.warning("Could not parse date string: %s", date_str)
    return None


def get_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = "D",
) -> pd.DatetimeIndex:
    """Create a ``DatetimeIndex`` between two dates.

    String dates are parsed via :func:`parse_date_string`.

    Args:
        start_date: Start of the range (string or ``datetime``).
        end_date: End of the range (string or ``datetime``).
        freq: Pandas offset alias for the frequency.

    Returns:
        The resulting ``DatetimeIndex``.
    """
    if isinstance(start_date, str):
        start_date = parse_date_string(start_date)  # type: ignore[assignment]

    if isinstance(end_date, str):
        end_date = parse_date_string(end_date)  # type: ignore[assignment]

    return pd.date_range(start=start_date, end=end_date, freq=freq)


def add_business_days(date: datetime, n_days: int) -> datetime:
    """Add *n_days* business days (Monday -- Friday) to *date*.

    Negative values are handled by delegating to
    :func:`subtract_business_days`.

    Args:
        date: Starting date.
        n_days: Number of business days to add.

    Returns:
        The resulting ``datetime``.
    """
    if n_days < 0:
        return subtract_business_days(date, abs(n_days))

    result: datetime = date
    day_count: int = 0

    while day_count < n_days:
        result += timedelta(days=1)
        if result.weekday() < 5:
            day_count += 1

    return result


def subtract_business_days(date: datetime, n_days: int) -> datetime:
    """Subtract *n_days* business days from *date*.

    Negative values are handled by delegating to
    :func:`add_business_days`.

    Args:
        date: Starting date.
        n_days: Number of business days to subtract.

    Returns:
        The resulting ``datetime``.
    """
    if n_days < 0:
        return add_business_days(date, abs(n_days))

    result: datetime = date
    day_count: int = 0

    while day_count < n_days:
        result -= timedelta(days=1)
        if result.weekday() < 5:
            day_count += 1

    return result


def is_business_day(date: datetime) -> bool:
    """Return *True* if *date* falls on Monday through Friday.

    Args:
        date: The date to check.

    Returns:
        Whether the date is a weekday.
    """
    return date.weekday() < 5


def current_quarter(
    date: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    """Return the (start, end) boundaries of the calendar quarter.

    Args:
        date: Reference date.  Defaults to ``datetime.now()``.

    Returns:
        ``(quarter_start, quarter_end)`` tuple.
    """
    if date is None:
        date = datetime.now()

    quarter: int = (date.month - 1) // 3 + 1
    start_month: int = (quarter - 1) * 3 + 1

    start_date: datetime = datetime(date.year, start_month, 1)

    if quarter < 4:
        end_date: datetime = datetime(
            date.year, start_month + 3, 1
        ) - timedelta(days=1)
    else:
        end_date = datetime(date.year + 1, 1, 1) - timedelta(days=1)

    return start_date, end_date


# ======================================================================
# File and Path Helpers
# ======================================================================


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Create *directory* (and parents) if it does not already exist.

    Args:
        directory: The directory path.

    Returns:
        The directory as a ``Path`` object.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_latest_file(
    directory: Union[str, Path], pattern: str = "*"
) -> Optional[Path]:
    """Return the most recently modified file matching *pattern*.

    Args:
        directory: The directory to search.
        pattern: Glob pattern for matching files.

    Returns:
        ``Path`` to the latest file, or *None* if no files match.
    """
    directory = Path(directory)
    matching_files: list[Path] = sorted(
        directory.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    return matching_files[0] if matching_files else None


def get_file_hash(
    file_path: Union[str, Path], algorithm: str = "md5"
) -> str:
    """Compute the hex-digest hash of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm name accepted by :func:`hashlib.new`.

    Returns:
        Hexadecimal hash digest.
    """
    file_path = Path(file_path)
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def save_json(
    data: Any,
    file_path: Union[str, Path],
    pretty: bool = True,
) -> None:
    """Serialise *data* to a JSON file.

    The parent directory is created automatically if needed.

    Args:
        data: Serialisable data structure.
        file_path: Destination path.
        pretty: Indent the output for readability.
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)

    with open(file_path, "w") as fh:
        if pretty:
            json.dump(data, fh, indent=2, sort_keys=True)
        else:
            json.dump(data, fh)


def load_json(file_path: Union[str, Path]) -> Any:
    """Deserialise data from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The deserialised data.
    """
    file_path = Path(file_path)

    with open(file_path, "r") as fh:
        return json.load(fh)


# ======================================================================
# Data Processing Helpers
# ======================================================================


def detect_outliers(
    series: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using z-score, IQR, or percentile methods.

    Args:
        series: Numeric series to analyse.
        method: One of ``'zscore'``, ``'iqr'``, or ``'percentile'``.
        threshold: Method-specific sensitivity parameter.

    Returns:
        Boolean series where *True* marks an outlier.

    Raises:
        ValueError: If *method* is unrecognised.
    """
    if method == "zscore":
        z_scores: pd.Series = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold

    if method == "iqr":
        q1: float = float(series.quantile(0.25))
        q3: float = float(series.quantile(0.75))
        iqr: float = q3 - q1
        lower_bound: float = q1 - threshold * iqr
        upper_bound: float = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)

    if method == "percentile":
        lower_bound_p: float = float(series.quantile(threshold / 100))
        upper_bound_p: float = float(series.quantile(1 - threshold / 100))
        return (series < lower_bound_p) | (series > upper_bound_p)

    raise ValueError(f"Unknown outlier detection method: {method}")


def infer_frequency(time_index: pd.DatetimeIndex) -> str:
    """Best-effort frequency inference for a datetime index.

    When ``pd.infer_freq`` fails (e.g. irregular spacing) this function
    uses the modal inter-observation gap to pick the nearest standard
    pandas offset alias.

    Args:
        time_index: The datetime index to analyse.

    Returns:
        A pandas offset alias string, or ``'unknown'`` if inference fails.
    """
    if len(time_index) < 3:
        return "unknown"

    diff: pd.Series = pd.Series(time_index[1:]) - pd.Series(time_index[:-1])
    most_common_diff: pd.Timedelta = diff.mode()[0]
    days: int = most_common_diff.days

    if days == 1:
        return "D"
    if days == 7:
        return "W"
    if 28 <= days <= 31:
        return "M"
    if 90 <= days <= 92:
        return "Q"
    if 365 <= days <= 366:
        return "Y"

    hours: int = most_common_diff.seconds // 3600
    if hours == 1:
        return "H"
    if hours == 24:
        return "D"

    return "unknown"


def moving_average(
    series: pd.Series, window: int, center: bool = False
) -> pd.Series:
    """Compute a simple moving average.

    Args:
        series: Input numeric series.
        window: Rolling window size.
        center: Whether to centre the window.

    Returns:
        The smoothed series.
    """
    return series.rolling(window=window, center=center).mean()


def exponential_smoothing(
    series: pd.Series, alpha: float = 0.3
) -> pd.Series:
    """Apply exponential weighted moving-average smoothing.

    Args:
        series: Input numeric series.
        alpha: Smoothing factor in ``(0, 1]``.

    Returns:
        The smoothed series.
    """
    return series.ewm(alpha=alpha).mean()


# ======================================================================
# Miscellaneous Helpers
# ======================================================================


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a random alphanumeric identifier.

    Args:
        prefix: Optional prefix separated by an underscore.
        length: Length of the random portion.

    Returns:
        The generated ID string.
    """
    chars: str = string.ascii_uppercase + string.ascii_lowercase + string.digits
    random_part: str = "".join(random.choice(chars) for _ in range(length))

    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def retry(
    func: Callable[[], _T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> _T:
    """Call *func* with exponential back-off retries.

    Args:
        func: A zero-argument callable to invoke.
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries (seconds).
        backoff: Multiplicative back-off factor.
        exceptions: Exception types that trigger a retry.

    Returns:
        The return value of *func* on the first successful call.

    Raises:
        Exception: Re-raises the last exception if all attempts fail.
    """
    attempt: int = 0
    while attempt < max_attempts:
        try:
            return func()
        except exceptions as exc:
            attempt += 1
            if attempt == max_attempts:
                raise

            sleep_time: float = delay * (backoff ** (attempt - 1))
            logger.warning(
                "Attempt %d failed: %s. Retrying in %.2f seconds...",
                attempt,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)

    # This line is technically unreachable, but keeps type-checkers happy.
    raise RuntimeError("retry exhausted without raising")  # pragma: no cover


def memoize(func: _F) -> _F:
    """Decorator that caches function results keyed by arguments.

    Args:
        func: The function to memoize.

    Returns:
        A wrapper that returns cached results for repeated calls.
    """
    cache: Dict[Tuple[Any, ...], Any] = {}

    @wraps(func)
    def memoized(*args: Any, **kwargs: Any) -> Any:
        key_args: Tuple[Any, ...] = tuple(args)
        key_kwargs: Tuple[Tuple[str, Any], ...] = tuple(sorted(kwargs.items()))
        key: Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]] = (
            key_args,
            key_kwargs,
        )

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized  # type: ignore[return-value]
