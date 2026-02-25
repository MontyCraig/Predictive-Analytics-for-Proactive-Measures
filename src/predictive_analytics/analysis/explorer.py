"""Exploratory data analysis for time-series data.

This module provides the :class:`TimeSeriesExplorer` class which bundles
visualisation helpers, statistical tests, and a convenience
:meth:`~TimeSeriesExplorer.run_full_analysis` method that produces a
complete EDA report.

Key changes from the legacy module:

* All ``print()`` calls in :meth:`test_stationarity` have been replaced
  with ``logger.info()`` for consistent observability.
* Full type annotations on every method (including return types).
* The helper :func:`load_data` is included for standalone usage.

Typical usage::

    from predictive_analytics.analysis.explorer import (
        TimeSeriesExplorer,
        load_data,
    )

    df = load_data("data/preprocessed/MSFT_preprocessed.csv")
    explorer = TimeSeriesExplorer(df)
    explorer.run_full_analysis(output_dir="output/eda")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from predictive_analytics.config.logging import setup_logging
from predictive_analytics.exceptions import DataNotLoadedError

__all__: list[str] = [
    "TimeSeriesExplorer",
    "load_data",
]

# Initialise module logger.
_logger = setup_logging()

# Use seaborn style for better visualizations.
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------


class TimeSeriesExplorer:
    """Explorer for time-series data with visualisation capabilities.

    Provides methods for plotting trends, seasonal decomposition,
    autocorrelation analysis, stationarity testing, distribution
    inspection, and correlation analysis.

    Args:
        data: A :class:`~pandas.DataFrame` with a
            :class:`~pandas.DatetimeIndex`.
        target_column: The column to focus analysis on.  Defaults to
            ``"close"``.

    Raises:
        DataNotLoadedError: If *data* is ``None`` or empty.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
    ) -> None:
        if data is None or data.empty:
            raise DataNotLoadedError(
                "Cannot initialise TimeSeriesExplorer with empty or "
                "None data."
            )

        self.data: pd.DataFrame = data
        self.target_column: str = target_column

        # Validate / coerce index.
        if not isinstance(data.index, pd.DatetimeIndex):
            _logger.warning(
                "Data index is not a DatetimeIndex. Converting..."
            )
            self.data.index = pd.to_datetime(self.data.index)

        _logger.info(
            "Time Series Explorer initialized with %d data points",
            len(data),
        )

    # ------------------------------------------------------------------ #
    # Visualisation methods
    # ------------------------------------------------------------------ #

    def plot_time_series(
        self,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Plot one or more columns over time.

        Args:
            columns: Column names to plot.  Defaults to
                ``[self.target_column]``.
        """
        if columns is None:
            columns = [self.target_column]

        plt.figure(figsize=(12, 6))

        for column in columns:
            if column in self.data.columns:
                plt.plot(
                    self.data.index, self.data[column], label=column
                )
            else:
                _logger.warning("Column %s not found in data", column)

        plt.title(f"Time Series Plot for {', '.join(columns)}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_seasonal_decomposition(
        self,
        period: int = 30,
        model: str = "additive",
    ) -> None:
        """Plot seasonal decomposition of the target column.

        Args:
            period: Period for the seasonal component.
            model: ``"additive"`` or ``"multiplicative"``.
        """
        _logger.info(
            "Performing %s seasonal decomposition with period %d",
            model,
            period,
        )

        try:
            result = seasonal_decompose(
                self.data[self.target_column],
                model=model,
                period=period,
            )

            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

            axes[0].plot(result.observed)
            axes[0].set_title("Observed")

            axes[1].plot(result.trend)
            axes[1].set_title("Trend")

            axes[2].plot(result.seasonal)
            axes[2].set_title("Seasonal")

            axes[3].plot(result.resid)
            axes[3].set_title("Residual")

            plt.tight_layout()
            plt.show()

        except Exception as exc:
            _logger.error(
                "Error in seasonal decomposition: %s", exc
            )
            raise

    def plot_acf_pacf(self, lags: int = 40) -> None:
        """Plot autocorrelation and partial autocorrelation functions.

        Args:
            lags: Number of lag orders to include.
        """
        _logger.info("Plotting ACF and PACF with %d lags", lags)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        plot_acf(
            self.data[self.target_column], lags=lags, ax=axes[0]
        )
        axes[0].set_title(
            f"Autocorrelation Function (ACF) for {self.target_column}"
        )

        plot_pacf(
            self.data[self.target_column], lags=lags, ax=axes[1]
        )
        axes[1].set_title(
            f"Partial Autocorrelation Function (PACF) for "
            f"{self.target_column}"
        )

        plt.tight_layout()
        plt.show()

    def test_stationarity(
        self,
    ) -> tuple[bool, float, dict[str, float]]:
        """Test stationarity of the target column using the ADF test.

        Performs the Augmented Dickey-Fuller test and logs the results.

        Returns:
            A 3-tuple of:

            * **is_stationary** (``bool``): ``True`` if the null
              hypothesis of a unit root is rejected at the 5 % level.
            * **p_value** (``float``): The test p-value.
            * **critical_values** (``dict[str, float]``): Mapping of
              significance level labels to critical values.
        """
        _logger.info(
            "Testing stationarity of %s", self.target_column
        )

        result = adfuller(self.data[self.target_column].dropna())
        test_statistic: float = result[0]
        p_value: float = result[1]
        critical_values: dict[str, float] = result[4]

        _logger.info(
            "Augmented Dickey-Fuller Test for %s",
            self.target_column,
        )
        _logger.info("Test Statistic: %.4f", test_statistic)
        _logger.info("p-value: %.4f", p_value)
        _logger.info("Critical Values:")
        for key, value in critical_values.items():
            _logger.info("  %s: %.4f", key, value)

        is_stationary: bool = p_value < 0.05
        if is_stationary:
            _logger.info(
                "Result: The time series is stationary (reject H0)"
            )
        else:
            _logger.info(
                "Result: The time series is non-stationary "
                "(fail to reject H0)"
            )

        return is_stationary, p_value, critical_values

    def plot_rolling_statistics(self, window: int = 20) -> None:
        """Plot rolling mean and standard deviation.

        Args:
            window: Window size (number of observations) for the
                rolling calculations.
        """
        _logger.info(
            "Plotting rolling statistics with window %d", window
        )

        rolling_mean: pd.Series = (
            self.data[self.target_column].rolling(window=window).mean()
        )
        rolling_std: pd.Series = (
            self.data[self.target_column].rolling(window=window).std()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data.index,
            self.data[self.target_column],
            label=self.target_column,
        )
        plt.plot(
            rolling_mean.index,
            rolling_mean,
            label=f"{window}-day Rolling Mean",
        )
        plt.plot(
            rolling_std.index,
            rolling_std,
            label=f"{window}-day Rolling Std",
        )
        plt.title(f"Rolling Statistics for {self.target_column}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_distribution(self) -> None:
        """Plot histogram with KDE and a Q-Q plot for the target column."""
        plt.figure(figsize=(12, 6))

        # Histogram.
        plt.subplot(1, 2, 1)
        sns.histplot(self.data[self.target_column], kde=True)
        plt.title(f"Distribution of {self.target_column}")

        # Q-Q Plot.
        plt.subplot(1, 2, 2)
        scipy_stats.probplot(
            self.data[self.target_column], dist="norm", plot=plt
        )
        plt.title("Q-Q Plot")

        plt.tight_layout()
        plt.show()

    def plot_box_plots(self, by: str = "month") -> None:
        """Plot box plots grouped by a time-period column.

        Args:
            by: Column to group by.  Common values are ``"month"``,
                ``"quarter"``, ``"year"``, and ``"day_of_week"``.  If
                the column does not yet exist in the data it will be
                derived from the index.
        """
        _calendar_derivations: dict[str, str] = {
            "day_of_week": "dayofweek",
            "month": "month",
            "quarter": "quarter",
            "year": "year",
        }

        if by not in self.data.columns and by in _calendar_derivations:
            _logger.warning(
                "Column %s not found in data. Adding it...", by
            )
            self.data[by] = getattr(
                self.data.index, _calendar_derivations[by]
            )

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=by, y=self.target_column, data=self.data)
        plt.title(f"Box Plot of {self.target_column} by {by}")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(
        self,
        columns: Optional[list[str]] = None,
    ) -> None:
        """Plot a Pearson correlation heatmap.

        Args:
            columns: Columns to include.  Defaults to all numeric
                columns in the DataFrame.
        """
        if columns is None:
            numeric_cols: list[str] = (
                self.data.select_dtypes(include=[np.number])
                .columns.tolist()
            )
            columns = numeric_cols

        corr_matrix: pd.DataFrame = self.data[columns].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_lag_scatter(self, max_lag: int = 10) -> None:
        """Plot scatter plots between the series and its lags.

        Args:
            max_lag: Maximum lag order to include.
        """
        # Create lag columns if they do not exist.
        for lag in range(1, max_lag + 1):
            lag_col: str = f"lag_{lag}"
            if lag_col not in self.data.columns:
                self.data[lag_col] = self.data[self.target_column].shift(
                    lag
                )

        # Create scatter plots.
        n_rows: int = (max_lag // 2) + (max_lag % 2)
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(14, max_lag * 2),
        )
        axes_flat = axes.flatten()

        for lag in range(1, max_lag + 1):
            lag_col = f"lag_{lag}"
            ax = axes_flat[lag - 1]
            ax.scatter(
                self.data[lag_col],
                self.data[self.target_column],
                alpha=0.5,
            )
            ax.set_title(f"Lag {lag} vs {self.target_column}")
            ax.set_xlabel(f"Lag {lag}")
            ax.set_ylabel(self.target_column)

        plt.tight_layout()
        plt.show()

    def run_full_analysis(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Run a complete exploratory analysis with all plots.

        When *output_dir* is provided each figure is saved as a PNG
        and the matplotlib figure is closed to free memory.  Otherwise
        figures are displayed interactively.

        Args:
            output_dir: Directory to save plot images.  Created
                automatically if it does not exist.
        """
        _logger.info("Starting full exploratory analysis")

        save_mode: bool = output_dir is not None
        if save_mode:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            _logger.info("Saving plots to %s", output_dir)

        # Time series plot.
        self.plot_time_series()
        if save_mode:
            plt.savefig(output_dir / "time_series_plot.png")
            plt.close()

        # Seasonal decomposition.
        self.plot_seasonal_decomposition()
        if save_mode:
            plt.savefig(output_dir / "seasonal_decomposition.png")
            plt.close()

        # ACF and PACF.
        self.plot_acf_pacf()
        if save_mode:
            plt.savefig(output_dir / "acf_pacf.png")
            plt.close()

        # Stationarity test.
        self.test_stationarity()

        # Rolling statistics.
        self.plot_rolling_statistics()
        if save_mode:
            plt.savefig(output_dir / "rolling_statistics.png")
            plt.close()

        # Distribution.
        self.plot_distribution()
        if save_mode:
            plt.savefig(output_dir / "distribution.png")
            plt.close()

        # Box plots by month.
        self.plot_box_plots()
        if save_mode:
            plt.savefig(output_dir / "box_plots_month.png")
            plt.close()

        # Correlation heatmap.
        self.plot_heatmap()
        if save_mode:
            plt.savefig(output_dir / "correlation_heatmap.png")
            plt.close()

        # Lag scatter plots.
        self.plot_lag_scatter()
        if save_mode:
            plt.savefig(output_dir / "lag_scatter.png")
            plt.close()

        _logger.info("Full exploratory analysis completed")


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load time-series data from a CSV file.

    The first column is used as the index and is parsed as dates.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A :class:`~pandas.DataFrame` with a
        :class:`~pandas.DatetimeIndex`.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        DataNotLoadedError: If the file cannot be read for any other
            reason.
    """
    resolved: Path = Path(file_path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    _logger.info("Loading data from %s", resolved)

    try:
        df: pd.DataFrame = pd.read_csv(
            resolved, index_col=0, parse_dates=True
        )
        _logger.info(
            "Loaded %d records from %s", len(df), resolved
        )
        return df
    except Exception as exc:
        _logger.error(
            "Error loading data from %s: %s", resolved, exc
        )
        raise DataNotLoadedError(
            f"Failed to load data from {resolved}: {exc}"
        ) from exc
