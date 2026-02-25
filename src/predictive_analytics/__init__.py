"""Predictive Analytics - Enterprise-grade time series forecasting and disruption prediction."""

from __future__ import annotations

from predictive_analytics.analysis.explorer import TimeSeriesExplorer
from predictive_analytics.collection.client import AlphaVantageClient
from predictive_analytics.config.settings import AppConfig
from predictive_analytics.disruption.analyzer import DisruptionAnalyzer
from predictive_analytics.modeling.forecaster import TimeSeriesForecaster
from predictive_analytics.modeling.trainer import ModelTrainer

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "AlphaVantageClient",
    "TimeSeriesExplorer",
    "ModelTrainer",
    "TimeSeriesForecaster",
    "DisruptionAnalyzer",
]
