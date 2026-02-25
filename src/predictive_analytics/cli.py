"""Typer-based CLI for the predictive-analytics pipeline.

Replaces the legacy ``argparse``-driven ``main.py`` with a modern CLI that
supports the same pipeline stages: collection, EDA, training, forecasting,
and disruption analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="predictive-analytics",
    help="Enterprise-grade time series forecasting and disruption prediction framework.",
    add_completion=False,
    no_args_is_help=True,
)


def _ensure_directory(path: Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Parameters
    ----------
    path:
        Target directory path.

    Returns
    -------
    Path
        The same *path* after ensuring it exists on disk.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_output_dir(output_dir: Optional[str], symbol: str) -> Path:
    """Determine and create the output directory for a pipeline run.

    Parameters
    ----------
    output_dir:
        Explicit output directory supplied by the user, or ``None`` to use the
        default ``./output/<symbol>`` layout.
    symbol:
        Stock ticker symbol used to derive the default path.

    Returns
    -------
    Path
        Resolved (and created) output directory.
    """
    if output_dir is not None:
        base = Path(output_dir)
    else:
        base = Path("output") / symbol.upper()
    return _ensure_directory(base)


def _run_collection(
    symbol: str,
    api_key: Optional[str],
    output_path: Path,
    config: object,
) -> None:
    """Execute the data-collection and preprocessing stage.

    Parameters
    ----------
    symbol:
        Ticker symbol to collect data for.
    api_key:
        Alpha Vantage API key override (``None`` to use config/env).
    output_path:
        Directory to write collected data files.
    config:
        Application configuration object.
    """
    from predictive_analytics.collection.preprocessing import collect_and_preprocess_data

    console.rule("[bold blue]Stage 1: Data Collection & Preprocessing")
    typer.echo(f"Collecting data for symbol: {symbol}")
    collect_and_preprocess_data(
        symbol=symbol,
        api_key=api_key,
        output_dir=output_path,
        config=config,
    )
    typer.echo("Data collection complete.")


def _run_eda(
    symbol: str,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Execute the Exploratory Data Analysis stage.

    Parameters
    ----------
    symbol:
        Ticker symbol whose data will be explored.
    output_path:
        Directory containing preprocessed data and for saving plots.
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    from predictive_analytics.analysis.explorer import TimeSeriesExplorer

    console.rule("[bold green]Stage 2: Exploratory Data Analysis")
    typer.echo("Running exploratory data analysis ...")
    explorer = TimeSeriesExplorer(
        symbol=symbol,
        data_dir=output_path,
        show_plots=show_plots,
    )
    explorer.run()
    typer.echo("EDA complete.")


def _run_training(
    symbol: str,
    output_path: Path,
) -> None:
    """Execute the model training and evaluation stage.

    Parameters
    ----------
    symbol:
        Ticker symbol whose data will be used for training.
    output_path:
        Directory containing preprocessed data and for saving model artefacts.
    """
    from predictive_analytics.modeling.trainer import train_and_evaluate_model

    console.rule("[bold yellow]Stage 3: Model Training & Evaluation")
    typer.echo("Training models ...")
    train_and_evaluate_model(
        symbol=symbol,
        data_dir=output_path,
    )
    typer.echo("Model training complete.")


def _run_forecasting(
    symbol: str,
    output_path: Path,
    forecast_steps: int,
    show_plots: bool,
) -> None:
    """Execute the forecasting stage.

    Parameters
    ----------
    symbol:
        Ticker symbol to forecast.
    output_path:
        Directory containing trained model artefacts.
    forecast_steps:
        Number of time steps to forecast into the future.
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    from predictive_analytics.modeling.forecaster import generate_forecast

    console.rule("[bold magenta]Stage 4: Forecasting")
    typer.echo(f"Generating {forecast_steps}-step forecast ...")
    generate_forecast(
        symbol=symbol,
        data_dir=output_path,
        steps=forecast_steps,
        show_plots=show_plots,
    )
    typer.echo("Forecasting complete.")


def _run_disruptions(
    symbol: str,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Execute the disruption / anomaly analysis stage.

    Parameters
    ----------
    symbol:
        Ticker symbol whose data will be analysed for disruptions.
    output_path:
        Directory containing preprocessed data and for saving reports.
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    from predictive_analytics.disruption.analyzer import analyze_disruptions

    console.rule("[bold red]Stage 5: Disruption Analysis")
    typer.echo("Analysing disruptions ...")
    analyze_disruptions(
        symbol=symbol,
        data_dir=output_path,
        show_plots=show_plots,
    )
    typer.echo("Disruption analysis complete.")


@app.command()
def run(
    symbol: str = typer.Argument(
        "MSFT",
        help="Stock ticker symbol to analyse.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="ALPHA_VANTAGE_API_KEY",
        help="Alpha Vantage API key. Falls back to config / environment variable.",
    ),
    env_file: Optional[str] = typer.Option(
        None,
        "--env-file",
        help="Path to a .env file for configuration overrides.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Root output directory.  Defaults to ./output/<SYMBOL>.",
    ),
    skip_collection: bool = typer.Option(
        False,
        "--skip-collection",
        help="Skip the data-collection stage.",
    ),
    skip_eda: bool = typer.Option(
        False,
        "--skip-eda",
        help="Skip exploratory data analysis.",
    ),
    skip_training: bool = typer.Option(
        False,
        "--skip-training",
        help="Skip model training and evaluation.",
    ),
    skip_forecasting: bool = typer.Option(
        False,
        "--skip-forecasting",
        help="Skip forecast generation.",
    ),
    skip_disruptions: bool = typer.Option(
        False,
        "--skip-disruptions",
        help="Skip disruption / anomaly analysis.",
    ),
    show_plots: bool = typer.Option(
        False,
        "--show-plots",
        help="Display interactive matplotlib plot windows.",
    ),
    forecast_steps: int = typer.Option(
        30,
        "--forecast-steps",
        min=1,
        help="Number of time steps to forecast.",
    ),
) -> None:
    """Run the full predictive-analytics pipeline for a given stock symbol."""
    from predictive_analytics.config.logging import setup_logging
    from predictive_analytics.config.settings import get_config

    setup_logging()
    logger.info("Starting predictive-analytics pipeline for %s", symbol)

    try:
        config = get_config(env_file)
        output_path = _resolve_output_dir(output_dir, symbol)

        if not skip_collection:
            _run_collection(symbol, api_key, output_path, config)

        if not skip_eda:
            _run_eda(symbol, output_path, show_plots)

        if not skip_training:
            _run_training(symbol, output_path)

        if not skip_forecasting:
            _run_forecasting(symbol, output_path, forecast_steps, show_plots)

        if not skip_disruptions:
            _run_disruptions(symbol, output_path, show_plots)

        console.rule("[bold green]Pipeline Complete")
        typer.echo(f"All results saved to: {output_path}")
    except KeyboardInterrupt:
        typer.echo("\nPipeline interrupted by user.", err=True)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed")
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
