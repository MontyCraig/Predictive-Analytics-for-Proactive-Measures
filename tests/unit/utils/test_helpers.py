"""Comprehensive tests for predictive_analytics.utils.helpers module.

Covers all 23 exported functions: string formatting, datetime helpers,
file/path helpers, data processing, and miscellaneous (generate_id, retry, memoize).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from predictive_analytics.utils.helpers import (
    add_business_days,
    current_quarter,
    detect_outliers,
    ensure_directory,
    exponential_smoothing,
    format_number,
    format_percent,
    generate_id,
    get_date_range,
    get_file_hash,
    get_latest_file,
    infer_frequency,
    is_business_day,
    load_json,
    memoize,
    moving_average,
    parse_date_string,
    retry,
    save_json,
    snake_to_camel,
    snake_to_title,
    subtract_business_days,
    truncate_string,
)

# ======================================================================
# String Formatting Functions
# ======================================================================


class TestSnakeToCamel:
    def test_single_word(self) -> None:
        assert snake_to_camel("hello") == "hello"

    def test_two_words(self) -> None:
        assert snake_to_camel("hello_world") == "helloWorld"

    def test_multiple_words(self) -> None:
        assert snake_to_camel("my_long_variable_name") == "myLongVariableName"

    def test_empty_string(self) -> None:
        assert snake_to_camel("") == ""

    def test_already_single(self) -> None:
        assert snake_to_camel("x") == "x"


class TestSnakeToTitle:
    def test_single_word(self) -> None:
        assert snake_to_title("hello") == "Hello"

    def test_two_words(self) -> None:
        assert snake_to_title("hello_world") == "Hello World"

    def test_multiple_words(self) -> None:
        assert snake_to_title("my_long_name") == "My Long Name"

    def test_empty_string(self) -> None:
        assert snake_to_title("") == ""


class TestTruncateString:
    def test_short_string_unchanged(self) -> None:
        assert truncate_string("hello", max_length=50) == "hello"

    def test_exact_length_unchanged(self) -> None:
        assert truncate_string("hello", max_length=5) == "hello"

    def test_truncation_with_default_suffix(self) -> None:
        result = truncate_string("hello world", max_length=8)
        assert result == "hello..."
        assert len(result) == 8

    def test_truncation_with_custom_suffix(self) -> None:
        result = truncate_string("hello world", max_length=9, suffix="~~")
        assert result == "hello w~~"
        assert len(result) == 9

    def test_very_short_max_length(self) -> None:
        result = truncate_string("hello world", max_length=3)
        assert result == "..."
        assert len(result) == 3


class TestFormatNumber:
    def test_default_precision_with_separator(self) -> None:
        assert format_number(1234567.89) == "1,234,567.89"

    def test_custom_precision(self) -> None:
        assert format_number(3.14159, precision=4) == "3.1416"

    def test_no_thousands_separator(self) -> None:
        assert format_number(1234.5, use_thousands_separator=False) == "1234.50"

    def test_zero(self) -> None:
        assert format_number(0) == "0.00"

    def test_negative(self) -> None:
        assert format_number(-1234.5, precision=1) == "-1,234.5"


class TestFormatPercent:
    def test_default_precision(self) -> None:
        assert format_percent(0.1) == "10.00%"

    def test_custom_precision(self) -> None:
        assert format_percent(0.12345, precision=1) == "12.3%"

    def test_zero_percent(self) -> None:
        assert format_percent(0) == "0.00%"

    def test_full_percent(self) -> None:
        assert format_percent(1.0) == "100.00%"


# ======================================================================
# DateTime Helper Functions
# ======================================================================


class TestParseDateString:
    def test_iso_format(self) -> None:
        result = parse_date_string("2024-01-15")
        assert result == datetime(2024, 1, 15)

    def test_slash_format(self) -> None:
        result = parse_date_string("2024/01/15")
        assert result == datetime(2024, 1, 15)

    def test_day_first_dash(self) -> None:
        result = parse_date_string("15-01-2024")
        assert result == datetime(2024, 1, 15)

    def test_day_first_slash(self) -> None:
        result = parse_date_string("15/01/2024")
        assert result == datetime(2024, 1, 15)

    def test_datetime_format(self) -> None:
        result = parse_date_string("2024-01-15 13:30:00")
        assert result == datetime(2024, 1, 15, 13, 30, 0)

    def test_slash_datetime_format(self) -> None:
        result = parse_date_string("2024/01/15 13:30:00")
        assert result == datetime(2024, 1, 15, 13, 30, 0)

    def test_invalid_format_returns_none(self) -> None:
        result = parse_date_string("not-a-date")
        assert result is None

    def test_custom_formats(self) -> None:
        result = parse_date_string("Jan 15, 2024", formats=["%b %d, %Y"])
        assert result == datetime(2024, 1, 15)

    def test_custom_formats_no_match(self) -> None:
        result = parse_date_string("2024-01-15", formats=["%b %d, %Y"])
        assert result is None


class TestGetDateRange:
    def test_string_dates(self) -> None:
        result = get_date_range("2024-01-01", "2024-01-05", freq="D")
        assert len(result) == 5
        assert isinstance(result, pd.DatetimeIndex)

    def test_datetime_dates(self) -> None:
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        result = get_date_range(start, end, freq="D")
        assert len(result) == 10

    def test_weekly_frequency(self) -> None:
        result = get_date_range("2024-01-01", "2024-02-01", freq="W")
        assert len(result) >= 4


class TestAddBusinessDays:
    def test_add_positive_days(self) -> None:
        # Monday 2024-01-15 + 5 business days = Monday 2024-01-22
        start = datetime(2024, 1, 15)
        result = add_business_days(start, 5)
        assert result == datetime(2024, 1, 22)

    def test_add_zero_days(self) -> None:
        start = datetime(2024, 1, 15)
        result = add_business_days(start, 0)
        assert result == start

    def test_add_skips_weekends(self) -> None:
        # Friday 2024-01-19 + 1 business day = Monday 2024-01-22
        start = datetime(2024, 1, 19)
        result = add_business_days(start, 1)
        assert result == datetime(2024, 1, 22)

    def test_negative_delegates_to_subtract(self) -> None:
        start = datetime(2024, 1, 22)
        result = add_business_days(start, -5)
        assert result == datetime(2024, 1, 15)


class TestSubtractBusinessDays:
    def test_subtract_positive_days(self) -> None:
        # Monday 2024-01-22 - 5 business days = Monday 2024-01-15
        start = datetime(2024, 1, 22)
        result = subtract_business_days(start, 5)
        assert result == datetime(2024, 1, 15)

    def test_subtract_zero_days(self) -> None:
        start = datetime(2024, 1, 15)
        result = subtract_business_days(start, 0)
        assert result == start

    def test_subtract_skips_weekends(self) -> None:
        # Monday 2024-01-22 - 1 = Friday 2024-01-19
        start = datetime(2024, 1, 22)
        result = subtract_business_days(start, 1)
        assert result == datetime(2024, 1, 19)

    def test_negative_delegates_to_add(self) -> None:
        start = datetime(2024, 1, 15)
        result = subtract_business_days(start, -5)
        assert result == datetime(2024, 1, 22)


class TestIsBusinessDay:
    def test_weekday_is_business_day(self) -> None:
        # Monday
        assert is_business_day(datetime(2024, 1, 15)) is True
        # Friday
        assert is_business_day(datetime(2024, 1, 19)) is True

    def test_weekend_is_not_business_day(self) -> None:
        # Saturday
        assert is_business_day(datetime(2024, 1, 20)) is False
        # Sunday
        assert is_business_day(datetime(2024, 1, 21)) is False


class TestCurrentQuarter:
    def test_q1(self) -> None:
        start, end = current_quarter(datetime(2024, 2, 15))
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 3, 31)

    def test_q2(self) -> None:
        start, end = current_quarter(datetime(2024, 5, 1))
        assert start == datetime(2024, 4, 1)
        assert end == datetime(2024, 6, 30)

    def test_q3(self) -> None:
        start, end = current_quarter(datetime(2024, 8, 20))
        assert start == datetime(2024, 7, 1)
        assert end == datetime(2024, 9, 30)

    def test_q4(self) -> None:
        start, end = current_quarter(datetime(2024, 12, 25))
        assert start == datetime(2024, 10, 1)
        assert end == datetime(2024, 12, 31)

    def test_default_uses_now(self) -> None:
        start, end = current_quarter()
        now = datetime.now()
        quarter = (now.month - 1) // 3 + 1
        expected_start_month = (quarter - 1) * 3 + 1
        assert start.month == expected_start_month
        assert start.year == now.year


# ======================================================================
# File and Path Helpers
# ======================================================================


class TestEnsureDirectory:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_directory(new_dir)
        assert result.exists()
        assert result == new_dir

    def test_existing_directory(self, tmp_path: Path) -> None:
        result = ensure_directory(tmp_path)
        assert result == tmp_path

    def test_accepts_string(self, tmp_path: Path) -> None:
        new_dir = str(tmp_path / "strdir")
        result = ensure_directory(new_dir)
        assert isinstance(result, Path)
        assert result.exists()


class TestGetLatestFile:
    def test_returns_latest_file(self, tmp_path: Path) -> None:
        # Create files with known modification order
        f1 = tmp_path / "a.txt"
        f1.write_text("a")
        import time as _time

        _time.sleep(0.05)
        f2 = tmp_path / "b.txt"
        f2.write_text("b")

        result = get_latest_file(tmp_path, "*.txt")
        assert result == f2

    def test_returns_none_when_no_match(self, tmp_path: Path) -> None:
        result = get_latest_file(tmp_path, "*.xyz")
        assert result is None

    def test_default_pattern(self, tmp_path: Path) -> None:
        f = tmp_path / "file.dat"
        f.write_text("data")
        result = get_latest_file(tmp_path)
        assert result is not None

    def test_accepts_string_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("1,2")
        result = get_latest_file(str(tmp_path), "*.csv")
        assert result == f


class TestGetFileHash:
    def test_md5_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        content = b"hello world"
        f.write_bytes(content)
        result = get_file_hash(f)
        expected = hashlib.md5(content).hexdigest()
        assert result == expected

    def test_sha256_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        content = b"test content"
        f.write_bytes(content)
        result = get_file_hash(f, algorithm="sha256")
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("text")
        result = get_file_hash(str(f))
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 hex digest


class TestSaveJson:
    def test_save_pretty(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "output.json"
        save_json(data, file_path, pretty=True)
        content = file_path.read_text()
        loaded = json.loads(content)
        assert loaded == data
        assert "\n" in content  # Pretty-printed

    def test_save_compact(self, tmp_path: Path) -> None:
        data = {"a": 1}
        file_path = tmp_path / "compact.json"
        save_json(data, file_path, pretty=False)
        content = file_path.read_text()
        loaded = json.loads(content)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        file_path = tmp_path / "deep" / "nested" / "dir" / "data.json"
        save_json({"x": 1}, file_path)
        assert file_path.exists()

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        file_path = str(tmp_path / "str_path.json")
        save_json([1, 2, 3], file_path)
        assert Path(file_path).exists()


class TestLoadJson:
    def test_load_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "list": [1, 2, 3]}
        file_path = tmp_path / "data.json"
        file_path.write_text(json.dumps(data))
        result = load_json(file_path)
        assert result == data

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "data.json"
        file_path.write_text('{"a": 1}')
        result = load_json(str(file_path))
        assert result == {"a": 1}


# ======================================================================
# Data Processing Helpers
# ======================================================================


class TestDetectOutliers:
    def test_zscore_method(self) -> None:
        data = pd.Series([1, 2, 3, 4, 5, 100])
        result = detect_outliers(data, method="zscore", threshold=2.0)
        assert result.iloc[-1] is True or result.iloc[-1] == True  # noqa: E712
        assert result.iloc[0] == False  # noqa: E712

    def test_iqr_method(self) -> None:
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = detect_outliers(data, method="iqr", threshold=1.5)
        assert result.iloc[-1] == True  # noqa: E712

    def test_percentile_method(self) -> None:
        data = pd.Series(range(100))
        result = detect_outliers(data, method="percentile", threshold=5.0)
        # Values below 5th percentile or above 95th percentile
        assert result.any()

    def test_unknown_method_raises(self) -> None:
        data = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown outlier detection method"):
            detect_outliers(data, method="nonexistent")


class TestInferFrequency:
    def test_daily_frequency(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        assert infer_frequency(idx) == "D"

    def test_weekly_frequency(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="W")
        assert infer_frequency(idx) == "W"

    def test_monthly_frequency(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="MS")
        result = infer_frequency(idx)
        assert result == "M"

    def test_quarterly_frequency(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="QS")
        result = infer_frequency(idx)
        assert result == "Q"

    def test_yearly_frequency(self) -> None:
        idx = pd.date_range("2020-01-01", periods=5, freq="YS")
        result = infer_frequency(idx)
        assert result == "Y"

    def test_hourly_frequency(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        result = infer_frequency(idx)
        assert result == "H"

    def test_too_few_points_returns_unknown(self) -> None:
        idx = pd.DatetimeIndex([datetime(2024, 1, 1), datetime(2024, 1, 2)])
        assert infer_frequency(idx) == "unknown"

    def test_single_point_returns_unknown(self) -> None:
        idx = pd.DatetimeIndex([datetime(2024, 1, 1)])
        assert infer_frequency(idx) == "unknown"

    def test_irregular_frequency_returns_unknown(self) -> None:
        # 2-day gaps don't match any standard frequency
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
            datetime(2024, 1, 5),
            datetime(2024, 1, 7),
        ]
        idx = pd.DatetimeIndex(dates)
        result = infer_frequency(idx)
        assert result == "unknown"

    def test_24_hour_gap_returns_daily(self) -> None:
        """Cover the ``hours == 24`` branch (line 503) in infer_frequency.

        Pandas normalises 24 hours into ``Timedelta(days=1, seconds=0)``
        so this branch is unreachable with real data.  We exercise it
        by creating a DatetimeIndex whose diff Series has a mode() that
        returns a synthetic Timedelta with days=0 and seconds=86400.
        """
        idx = pd.date_range("2024-01-01", periods=5, freq="D")

        class _FakeTimedelta:
            """Looks like a Timedelta but with un-normalised 24 hours."""

            days = 0
            seconds = 86400

        # Compute the real diff so pd.Series is called normally, then
        # we swap the mode() return just before the comparisons happen.
        real_pd_series = pd.Series

        class _PatchedDiffSeries(pd.Series):
            """pd.Series subclass that overrides mode()."""

            def mode(self, dropna: bool = True) -> pd.Series:  # type: ignore[override]
                return pd.Series([_FakeTimedelta()])

        call_idx = {"n": 0}

        def _patched_series(*args: Any, **kwargs: Any) -> Any:
            call_idx["n"] += 1
            s = real_pd_series(*args, **kwargs)
            # The subtraction (call 3, the diff) is what we want to patch.
            # Calls 1 and 2 create the left/right index series.
            # But subtraction returns a new Series, so we can't intercept
            # it here.  Instead we patch at the __sub__ level.
            return s

        # Easier: the diff is created by subtracting two pd.Series.
        # We can patch pd.Series.__sub__ to return our special subclass.
        original_sub = pd.Series.__sub__

        def _patched_sub(self: pd.Series, other: Any) -> _PatchedDiffSeries:
            result = original_sub(self, other)
            return _PatchedDiffSeries(result)

        with patch.object(pd.Series, "__sub__", _patched_sub):
            result = infer_frequency(idx)
        assert result == "D"


class TestMovingAverage:
    def test_basic_moving_average(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        result = moving_average(s, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_centered_moving_average(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        result = moving_average(s, window=3, center=True)
        assert result.iloc[1] == pytest.approx(2.0)
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[3] == pytest.approx(4.0)


class TestExponentialSmoothing:
    def test_basic_smoothing(self) -> None:
        s = pd.Series([10, 20, 30, 40, 50])
        result = exponential_smoothing(s, alpha=0.5)
        assert len(result) == 5
        # First value should be 10 (only one observation)
        assert result.iloc[0] == pytest.approx(10.0)
        # Later values should be smoothed toward recent data
        assert result.iloc[-1] > result.iloc[0]

    def test_alpha_one_equals_original(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        result = exponential_smoothing(s, alpha=1.0)
        pd.testing.assert_series_equal(result, s.astype(float), check_names=False)


# ======================================================================
# Miscellaneous Helpers
# ======================================================================


class TestGenerateId:
    def test_default_length(self) -> None:
        result = generate_id()
        assert len(result) == 8
        assert result.isalnum()

    def test_custom_length(self) -> None:
        result = generate_id(length=16)
        assert len(result) == 16

    def test_with_prefix(self) -> None:
        result = generate_id(prefix="test")
        assert result.startswith("test_")
        assert len(result) == len("test_") + 8

    def test_without_prefix(self) -> None:
        result = generate_id(prefix="")
        assert "_" not in result

    def test_uniqueness(self) -> None:
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100  # All should be unique


class TestRetry:
    def test_succeeds_first_try(self) -> None:
        result = retry(lambda: 42, max_attempts=3)
        assert result == 42

    def test_succeeds_after_retries(self) -> None:
        call_count = {"n": 0}

        def flaky() -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "ok"

        with patch("predictive_analytics.utils.helpers.time.sleep"):
            result = retry(flaky, max_attempts=3, delay=0.01)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_after_max_attempts(self) -> None:
        def always_fail() -> None:
            raise RuntimeError("always fails")

        with patch("predictive_analytics.utils.helpers.time.sleep"):
            with pytest.raises(RuntimeError, match="always fails"):
                retry(always_fail, max_attempts=3, delay=0.01)

    def test_only_catches_specified_exceptions(self) -> None:
        def raise_type_error() -> None:
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            retry(
                raise_type_error,
                max_attempts=3,
                exceptions=(ValueError,),
            )

    def test_exponential_backoff(self) -> None:
        call_count = {"n": 0}

        def fail_twice() -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("nope")
            return "done"

        with patch("predictive_analytics.utils.helpers.time.sleep") as mock_sleep:
            retry(fail_twice, max_attempts=3, delay=1.0, backoff=2.0)

        # First retry: delay * backoff^0 = 1.0
        # Second retry: delay * backoff^1 = 2.0
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


class TestMemoize:
    def test_caches_result(self) -> None:
        call_count = {"n": 0}

        @memoize
        def expensive(x: int) -> int:
            call_count["n"] += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count["n"] == 1  # Called only once

    def test_different_args_not_cached(self) -> None:
        call_count = {"n": 0}

        @memoize
        def fn(x: int) -> int:
            call_count["n"] += 1
            return x + 1

        assert fn(1) == 2
        assert fn(2) == 3
        assert call_count["n"] == 2

    def test_kwargs_are_cached(self) -> None:
        call_count = {"n": 0}

        @memoize
        def fn(a: int, b: int = 0) -> int:
            call_count["n"] += 1
            return a + b

        assert fn(1, b=2) == 3
        assert fn(1, b=2) == 3
        assert call_count["n"] == 1

    def test_different_kwargs_not_cached(self) -> None:
        call_count = {"n": 0}

        @memoize
        def fn(a: int, b: int = 0) -> int:
            call_count["n"] += 1
            return a + b

        assert fn(1, b=2) == 3
        assert fn(1, b=3) == 4
        assert call_count["n"] == 2

    def test_preserves_function_name(self) -> None:
        @memoize
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"
