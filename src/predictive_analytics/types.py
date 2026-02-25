"""Shared type aliases used across the predictive-analytics package."""

from __future__ import annotations

from typing import TypeAlias

SARIMAOrder: TypeAlias = tuple[int, int, int]
SARIMASeasonalOrder: TypeAlias = tuple[int, int, int, int]
MetricsDict: TypeAlias = dict[str, float]
DisruptionReport: TypeAlias = dict[str, int | float | list[str]]
