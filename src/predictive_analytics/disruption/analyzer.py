"""Identify potential disruptions in time series forecasts.

Provides the ``DisruptionAnalyzer`` class which detects four categories of
disruption -- trend, volatility, level shift, and uncertainty -- in forecast
data and can produce consolidated reports and visualisations.

A convenience function :func:`analyze_disruptions` wraps the full
detect-report-save workflow for common use-cases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime as dt_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from predictive_analytics.config.logging import setup_logging
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.exceptions import DataNotLoadedError, DisruptionAnalysisError
from predictive_analytics.types import DisruptionReport

__all__: list[str] = ["DisruptionAnalyzer", "analyze_disruptions"]

logger: logging.Logger = setup_logging()


class DisruptionAnalyzer:
    """Detect and classify potential disruptions in time series forecasts.

    Supports four disruption categories:

    * **Trend** -- abnormal slope changes in a rolling window.
    * **Volatility** -- unusual rolling standard deviation.
    * **Level shift** -- sudden jumps between consecutive forecast values.
    * **Uncertainty** -- abnormally wide prediction intervals.

    Typical usage::

        analyzer = DisruptionAnalyzer(config, forecast_data=forecast_df)
        disruption_df = analyzer.identify_all_disruptions()
        report = analyzer.generate_disruption_report(disruption_df)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: AppConfig,
        forecast_data: Optional[pd.DataFrame] = None,
        historical_data: Optional[pd.DataFrame] = None,
        target_column: str = "close",
    ) -> None:
        """Initialise the disruption analyser.

        Args:
            config: Application configuration.
            forecast_data: DataFrame containing at least a ``forecast``
                column plus optional ``lower_bound`` / ``upper_bound``.
            historical_data: Historical time series DataFrame.
            target_column: Column name for the target variable in
                *historical_data*.
        """
        self.config: AppConfig = config
        self.forecast_data: Optional[pd.DataFrame] = forecast_data
        self.historical_data: Optional[pd.DataFrame] = historical_data
        self.target_column: str = target_column

        logger.info("DisruptionAnalyzer initialized")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_forecast(self, file_path: Union[str, Path]) -> None:
        """Load forecast data from a CSV file.

        Args:
            file_path: Path to the forecast CSV.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            DisruptionAnalysisError: If the file cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("Forecast file not found: %s", file_path)
            raise FileNotFoundError(f"Forecast file not found: {file_path}")

        logger.info("Loading forecast data from %s", file_path)

        try:
            self.forecast_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info("Loaded %d forecast records", len(self.forecast_data))
        except Exception as exc:
            logger.error("Error loading forecast data: %s", exc)
            raise DisruptionAnalysisError(
                f"Failed to load forecast data from {file_path}"
            ) from exc

    def load_historical_data(self, file_path: Union[str, Path]) -> None:
        """Load historical data from a CSV file.

        Args:
            file_path: Path to the historical data CSV.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            DisruptionAnalysisError: If the file cannot be parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error("Historical data file not found: %s", file_path)
            raise FileNotFoundError(f"Historical data file not found: {file_path}")

        logger.info("Loading historical data from %s", file_path)

        try:
            self.historical_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info("Loaded %d historical records", len(self.historical_data))
        except Exception as exc:
            logger.error("Error loading historical data: %s", exc)
            raise DisruptionAnalysisError(
                f"Failed to load historical data from {file_path}"
            ) from exc

    # ------------------------------------------------------------------
    # Individual disruption detectors
    # ------------------------------------------------------------------

    def identify_trend_disruptions(
        self, window_size: int = 5, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Identify abnormal trend changes using rolling linear regression.

        Args:
            window_size: Rolling window length for slope calculation.
            threshold: Z-score threshold above which a slope is flagged.

        Returns:
            Copy of the forecast data with ``rolling_slope``,
            ``slope_z_score`` and ``trend_disruption`` columns appended.

        Raises:
            DataNotLoadedError: If forecast data has not been loaded.
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise DataNotLoadedError("Forecast data not loaded. Please load forecast data first.")

        logger.info("Identifying trend disruptions with threshold %s", threshold)

        result_df: pd.DataFrame = self.forecast_data.copy()
        x: np.ndarray = np.arange(window_size)
        result_df["rolling_slope"] = np.nan

        for i in range(len(result_df) - window_size + 1):
            y: np.ndarray = result_df["forecast"].iloc[i : i + window_size].values
            slope: float
            slope, _, _, _, _ = stats.linregress(x, y)
            result_df.iloc[
                i + window_size - 1,
                result_df.columns.get_loc("rolling_slope"),
            ] = slope

        valid_slopes: pd.Series = result_df["rolling_slope"].dropna()
        slope_mean: float = float(valid_slopes.mean())
        slope_std: float = float(valid_slopes.std())

        if slope_std > 0:
            result_df["slope_z_score"] = np.abs(
                (result_df["rolling_slope"] - slope_mean) / slope_std
            )
            result_df["trend_disruption"] = result_df["slope_z_score"] > threshold

            n_disruptions: int = int(result_df["trend_disruption"].sum())
            logger.info("Detected %d trend disruptions", n_disruptions)
        else:
            logger.warning("Slope standard deviation is zero. No trend disruptions detected.")
            result_df["slope_z_score"] = 0.0
            result_df["trend_disruption"] = False

        return result_df

    def identify_volatility_disruptions(
        self, window_size: int = 10, threshold: float = 2.0
    ) -> pd.DataFrame:
        """Identify abnormal forecast volatility using rolling standard deviation.

        Args:
            window_size: Rolling window length for volatility calculation.
            threshold: Z-score threshold for flagging.

        Returns:
            Copy of the forecast data with ``rolling_volatility``,
            ``volatility_z_score`` and ``volatility_disruption`` columns
            appended.

        Raises:
            DataNotLoadedError: If forecast data has not been loaded.
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise DataNotLoadedError("Forecast data not loaded. Please load forecast data first.")

        logger.info("Identifying volatility disruptions with threshold %s", threshold)

        result_df: pd.DataFrame = self.forecast_data.copy()
        result_df["rolling_volatility"] = result_df["forecast"].rolling(window=window_size).std()

        valid_volatility: pd.Series = result_df["rolling_volatility"].dropna()
        vol_mean: float = float(valid_volatility.mean())
        vol_std: float = float(valid_volatility.std())

        if vol_std > 0:
            result_df["volatility_z_score"] = np.abs(
                (result_df["rolling_volatility"] - vol_mean) / vol_std
            )
            result_df["volatility_disruption"] = result_df["volatility_z_score"] > threshold

            n_disruptions: int = int(result_df["volatility_disruption"].sum())
            logger.info("Detected %d volatility disruptions", n_disruptions)
        else:
            logger.warning(
                "Volatility standard deviation is zero. " "No volatility disruptions detected."
            )
            result_df["volatility_z_score"] = 0.0
            result_df["volatility_disruption"] = False

        return result_df

    def identify_level_disruptions(self, threshold_std: float = 2.0) -> pd.DataFrame:
        """Identify sudden level shifts between consecutive forecast values.

        Args:
            threshold_std: Multiple of the standard deviation of
                consecutive differences used to flag level shifts.

        Returns:
            Copy of the forecast data with ``forecast_diff`` and
            ``level_disruption`` columns appended.

        Raises:
            DataNotLoadedError: If forecast data has not been loaded.
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise DataNotLoadedError("Forecast data not loaded. Please load forecast data first.")

        logger.info(
            "Identifying level disruptions with threshold %s std",
            threshold_std,
        )

        result_df: pd.DataFrame = self.forecast_data.copy()
        result_df["forecast_diff"] = result_df["forecast"].diff()

        diffs: pd.Series = result_df["forecast_diff"].dropna()
        diff_mean: float = float(diffs.mean())
        diff_std: float = float(diffs.std())

        threshold_value: float = diff_std * threshold_std
        result_df["level_disruption"] = (
            np.abs(result_df["forecast_diff"] - diff_mean) > threshold_value
        )

        n_disruptions: int = int(result_df["level_disruption"].sum())
        logger.info("Detected %d level disruptions", n_disruptions)

        return result_df

    def identify_uncertainty_disruptions(self, threshold: float = 2.0) -> pd.DataFrame:
        """Identify periods with abnormally wide prediction intervals.

        Args:
            threshold: Z-score threshold for flagging.

        Returns:
            Copy of the forecast data with ``interval_width``,
            ``uncertainty_z_score`` and ``uncertainty_disruption`` columns
            appended.

        Raises:
            DataNotLoadedError: If forecast data has not been loaded.
            DisruptionAnalysisError: If the forecast data lacks prediction
                interval columns.
        """
        if self.forecast_data is None:
            logger.error("Forecast data not loaded. Please load forecast data first.")
            raise DataNotLoadedError("Forecast data not loaded. Please load forecast data first.")

        required_cols: set[str] = {"lower_bound", "upper_bound"}
        if not required_cols.issubset(self.forecast_data.columns):
            logger.error("Forecast data does not contain prediction intervals.")
            raise DisruptionAnalysisError(
                "Forecast data does not contain prediction intervals "
                "(requires 'lower_bound' and 'upper_bound' columns)."
            )

        logger.info("Identifying uncertainty disruptions with threshold %s", threshold)

        result_df: pd.DataFrame = self.forecast_data.copy()
        result_df["interval_width"] = result_df["upper_bound"] - result_df["lower_bound"]

        width_mean: float = float(result_df["interval_width"].mean())
        width_std: float = float(result_df["interval_width"].std())

        if width_std > 0:
            result_df["uncertainty_z_score"] = np.abs(
                (result_df["interval_width"] - width_mean) / width_std
            )
            result_df["uncertainty_disruption"] = result_df["uncertainty_z_score"] > threshold

            n_disruptions: int = int(result_df["uncertainty_disruption"].sum())
            logger.info("Detected %d uncertainty disruptions", n_disruptions)
        else:
            logger.warning(
                "Interval width standard deviation is zero. "
                "No uncertainty disruptions detected."
            )
            result_df["uncertainty_z_score"] = 0.0
            result_df["uncertainty_disruption"] = False

        return result_df

    # ------------------------------------------------------------------
    # Consolidated detection
    # ------------------------------------------------------------------

    def identify_all_disruptions(self) -> pd.DataFrame:
        """Run all four disruption detectors and consolidate results.

        Returns:
            Copy of the forecast data with all disruption flags plus
            ``total_disruptions`` and ``significant_disruption`` columns.

        Raises:
            DataNotLoadedError: If forecast data has not been loaded.
        """
        logger.info("Identifying all types of disruptions")

        trend_df: pd.DataFrame = self.identify_trend_disruptions()
        volatility_df: pd.DataFrame = self.identify_volatility_disruptions()
        level_df: pd.DataFrame = self.identify_level_disruptions()
        uncertainty_df: pd.DataFrame = self.identify_uncertainty_disruptions()

        result_df: pd.DataFrame = self.forecast_data.copy()  # type: ignore[union-attr]
        result_df["trend_disruption"] = trend_df["trend_disruption"]
        result_df["volatility_disruption"] = volatility_df["volatility_disruption"]
        result_df["level_disruption"] = level_df["level_disruption"]
        result_df["uncertainty_disruption"] = uncertainty_df["uncertainty_disruption"]

        result_df["total_disruptions"] = (
            result_df["trend_disruption"].astype(int)
            + result_df["volatility_disruption"].astype(int)
            + result_df["level_disruption"].astype(int)
            + result_df["uncertainty_disruption"].astype(int)
        )

        result_df["significant_disruption"] = result_df["total_disruptions"] >= 2

        n_significant: int = int(result_df["significant_disruption"].sum())
        logger.info("Detected %d significant disruptions", n_significant)

        return result_df

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_disruptions(
        self,
        disruption_df: pd.DataFrame,
        historical_periods: int = 60,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot the forecast with disruption highlights.

        The upper subplot shows the forecast line, prediction intervals, and
        scatter points for significant disruptions.  The lower subplot shows
        a grouped bar chart of disruption types by date.

        Args:
            disruption_df: DataFrame produced by
                :meth:`identify_all_disruptions`.
            historical_periods: Number of trailing historical periods to
                include.
            save_path: When provided the figure is saved rather than shown.
        """
        if "significant_disruption" not in disruption_df.columns:
            logger.warning(
                "Disruption DataFrame does not contain " "significant_disruption column."
            )
            return

        plt.figure(figsize=(14, 10))

        # ---- Upper subplot: forecast + disruptions -----------------------
        if self.historical_data is not None:
            historical_data: pd.DataFrame = self.historical_data.tail(historical_periods)
            plt.subplot(2, 1, 1)
            plt.plot(
                historical_data.index,
                historical_data[self.target_column],
                label="Historical Data",
                color="blue",
            )

        plt.subplot(2, 1, 1)
        plt.plot(
            disruption_df.index,
            disruption_df["forecast"],
            label="Forecast",
            color="green",
            linestyle="-",
        )

        significant_points: pd.DataFrame = disruption_df[disruption_df["significant_disruption"]]
        if not significant_points.empty:
            plt.scatter(
                significant_points.index,
                significant_points["forecast"],
                color="red",
                s=100,
                label="Significant Disruptions",
                zorder=5,
            )

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

        # ---- Lower subplot: disruption type bars -------------------------
        plt.subplot(2, 1, 2)
        width: float = 0.2
        x: np.ndarray = np.arange(len(disruption_df))

        _bar_specs: list[tuple[str, float, str]] = [
            ("trend_disruption", -1.5 * width, "Trend Disruption"),
            ("volatility_disruption", -0.5 * width, "Volatility Disruption"),
            ("level_disruption", 0.5 * width, "Level Disruption"),
            ("uncertainty_disruption", 1.5 * width, "Uncertainty Disruption"),
        ]

        for col_name, offset, label in _bar_specs:
            if col_name in disruption_df.columns:
                plt.bar(
                    x + offset,
                    disruption_df[col_name].astype(int),
                    width=width,
                    label=label,
                )

        plt.xticks(
            x,
            [d.strftime("%Y-%m-%d") for d in disruption_df.index],
            rotation=45,
        )
        plt.title("Types of Disruptions by Date")
        plt.xlabel("Date")
        plt.ylabel("Disruption (0=No, 1=Yes)")
        plt.legend()
        plt.grid(True, axis="y")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info("Disruption plot saved to %s", save_path)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_disruption_report(self, disruption_df: pd.DataFrame) -> DisruptionReport:
        """Produce a summary report dictionary for the given disruptions.

        Args:
            disruption_df: DataFrame produced by
                :meth:`identify_all_disruptions`.

        Returns:
            A :class:`DisruptionReport` typed dictionary containing counts,
            percentages, and dates for each disruption category.
        """
        logger.info("Generating disruption report")

        def _col_sum(col: str) -> int:
            if col in disruption_df.columns:
                return int(disruption_df[col].sum())
            return 0

        trend_disruptions: int = _col_sum("trend_disruption")
        volatility_disruptions: int = _col_sum("volatility_disruption")
        level_disruptions: int = _col_sum("level_disruption")
        uncertainty_disruptions: int = _col_sum("uncertainty_disruption")
        significant_disruptions: int = _col_sum("significant_disruption")

        significant_dates: List[Any]
        if "significant_disruption" in disruption_df.columns:
            significant_dates = disruption_df[
                disruption_df["significant_disruption"]
            ].index.tolist()
        else:
            significant_dates = []

        total_periods: int = len(disruption_df)
        disruption_percentage: float = (
            (significant_disruptions / total_periods) * 100 if total_periods > 0 else 0.0
        )

        report: DisruptionReport = {
            "trend_disruptions": trend_disruptions,
            "volatility_disruptions": volatility_disruptions,
            "level_disruptions": level_disruptions,
            "uncertainty_disruptions": uncertainty_disruptions,
            "significant_disruptions": significant_disruptions,
            "total_forecast_periods": total_periods,
            "disruption_percentage": disruption_percentage,
            "significant_disruption_dates": significant_dates,
        }

        return report


# =====================================================================
# Convenience function
# =====================================================================


def analyze_disruptions(
    config: AppConfig,
    forecast_path: Optional[Union[str, Path]] = None,
    historical_data_path: Optional[Union[str, Path]] = None,
    target_column: str = "close",
    generate_new_forecast: bool = False,
    forecast_steps: int = 30,
    plot: bool = True,
    save_results: bool = True,
) -> Tuple[pd.DataFrame, DisruptionReport]:
    """Detect and report disruptions in a time series forecast.

    This is a high-level convenience wrapper that orchestrates forecast
    loading (or generation), disruption detection, reporting, and optional
    persistence in a single call.

    Args:
        config: Application configuration.
        forecast_path: Path to a pre-existing forecast CSV.
        historical_data_path: Path to historical data CSV.
        target_column: Target column name.
        generate_new_forecast: When *True* a fresh forecast is produced
            using :func:`~predictive_analytics.modeling.forecaster.generate_forecast`.
        forecast_steps: Number of forecast steps (only used when
            *generate_new_forecast* is *True*).
        plot: Whether to display / save visualisations.
        save_results: Whether to persist disruption data, report, and
            plot to disk.

    Returns:
        A ``(disruption_df, report)`` tuple.

    Raises:
        DisruptionAnalysisError: If analysis fails.
    """
    from predictive_analytics.modeling.forecaster import generate_forecast

    analyzer = DisruptionAnalyzer(config, target_column=target_column)

    # Load or generate forecast
    if generate_new_forecast or forecast_path is None:
        logger.info("Generating new forecast")
        forecast_df: pd.DataFrame = generate_forecast(
            config,
            steps=forecast_steps,
            target_column=target_column,
            plot=False,
            save_forecast=True,
        )
        analyzer.forecast_data = forecast_df
    else:
        logger.info("Loading forecast from %s", forecast_path)
        analyzer.load_forecast(forecast_path)

    # Load historical data
    if historical_data_path:
        analyzer.load_historical_data(historical_data_path)
    elif historical_data_path is None and analyzer.historical_data is None:
        preprocessed_dir: Path = Path(config.output_dir) / "preprocessed"
        if preprocessed_dir.exists():
            data_files: List[Path] = list(preprocessed_dir.glob("*_preprocessed.csv"))
            if data_files:
                data_path: Path = max(data_files, key=lambda p: p.stat().st_mtime)
                analyzer.load_historical_data(data_path)

    # Detect disruptions
    disruption_df: pd.DataFrame = analyzer.identify_all_disruptions()

    # Generate report
    report: DisruptionReport = analyzer.generate_disruption_report(disruption_df)

    logger.info("Disruption analysis summary:")
    logger.info("  Total forecast periods: %d", report["total_forecast_periods"])
    logger.info(
        "  Significant disruptions: %d (%.2f%%)",
        report["significant_disruptions"],
        report["disruption_percentage"],
    )
    logger.info("  Trend disruptions: %d", report["trend_disruptions"])
    logger.info("  Volatility disruptions: %d", report["volatility_disruptions"])
    logger.info("  Level disruptions: %d", report["level_disruptions"])
    logger.info("  Uncertainty disruptions: %d", report["uncertainty_disruptions"])

    if report["significant_disruption_dates"]:
        logger.info("Significant disruption dates:")
        for date in report["significant_disruption_dates"]:
            if hasattr(date, "strftime"):
                logger.info("  - %s", date.strftime("%Y-%m-%d"))
            else:
                logger.info("  - %s", str(date))

    # Plot
    if plot:
        analyzer.plot_disruptions(disruption_df)

    # Save results
    if save_results:
        disruptions_dir: Path = Path(config.output_dir) / "disruptions"
        disruptions_dir.mkdir(parents=True, exist_ok=True)

        timestamp: str = dt_datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save disruption data CSV
        disruption_path: Path = disruptions_dir / f"disruptions_{timestamp}.csv"
        disruption_df.to_csv(disruption_path)
        logger.info("Disruption data saved to %s", disruption_path)

        # Save plot
        if plot:
            plot_path: Path = disruptions_dir / f"disruptions_plot_{timestamp}.png"
            analyzer.plot_disruptions(disruption_df, save_path=plot_path)

        # Save JSON report (convert datetime objects for serialisation)
        report_copy: Dict[str, Any] = dict(report)
        report_copy["significant_disruption_dates"] = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            for d in report["significant_disruption_dates"]
        ]

        report_path: Path = disruptions_dir / f"disruption_report_{timestamp}.json"
        with open(report_path, "w") as fh:
            json.dump(report_copy, fh, indent=2)
        logger.info("Disruption report saved to %s", report_path)

    return disruption_df, report
