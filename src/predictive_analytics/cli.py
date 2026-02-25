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
    config: object,
    symbol: str,
) -> None:
    """Execute the data-collection and preprocessing stage.

    Parameters
    ----------
    config:
        Application configuration object (``AppConfig``).
    symbol:
        Ticker symbol to collect data for.
    """
    from predictive_analytics.collection.preprocessing import (  # noqa: C0415
        collect_and_preprocess_data,
    )

    console.rule("[bold blue]Stage 1: Data Collection & Preprocessing")
    typer.echo(f"Collecting data for symbol: {symbol}")
    collect_and_preprocess_data(config=config, symbol=symbol, save=True)  # type: ignore[arg-type]
    typer.echo("Data collection complete.")


def _run_eda(
    config: object,
    output_path: Path,
    show_plots: bool,
) -> None:
    """Execute the Exploratory Data Analysis stage.

    Parameters
    ----------
    config:
        Application configuration object (``AppConfig``).
    output_path:
        Directory containing preprocessed data and for saving plots.
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    import pandas as pd  # noqa: C0415

    from predictive_analytics.analysis.explorer import TimeSeriesExplorer  # noqa: C0415

    console.rule("[bold green]Stage 2: Exploratory Data Analysis")
    typer.echo("Running exploratory data analysis ...")

    # Load preprocessed data from the output directory.
    csv_files = list(output_path.glob("*preprocessed*.csv"))
    if not csv_files:
        typer.secho(
            "No preprocessed data found. Run collection first.",
            fg="yellow",
            err=True,
        )
        raise typer.Exit(1)

    data = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)

    explorer = TimeSeriesExplorer(data=data, target_column="close")
    save_dir: Optional[Path] = output_path / "eda" if not show_plots else None
    explorer.run_full_analysis(output_dir=save_dir)
    typer.echo("EDA complete.")


def _run_training(
    config: object,
    output_path: Path,
) -> None:
    """Execute the model training and evaluation stage.

    Parameters
    ----------
    config:
        Application configuration object (``AppConfig``).
    output_path:
        Directory containing preprocessed data and for saving model artefacts.
    """
    from predictive_analytics.modeling.trainer import train_and_evaluate_model  # noqa: C0415

    console.rule("[bold yellow]Stage 3: Model Training & Evaluation")
    typer.echo("Training models ...")
    train_and_evaluate_model(config=config)  # type: ignore[arg-type]
    typer.echo("Model training complete.")


def _run_forecasting(
    config: object,
    forecast_steps: int,
    show_plots: bool,
) -> None:
    """Execute the forecasting stage.

    Parameters
    ----------
    config:
        Application configuration object (``AppConfig``).
    forecast_steps:
        Number of time steps to forecast into the future.
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    from predictive_analytics.modeling.forecaster import generate_forecast  # noqa: C0415

    console.rule("[bold magenta]Stage 4: Forecasting")
    typer.echo(f"Generating {forecast_steps}-step forecast ...")
    generate_forecast(
        config=config,  # type: ignore[arg-type]
        steps=forecast_steps,
        plot=show_plots,
    )
    typer.echo("Forecasting complete.")


def _run_disruptions(
    config: object,
    show_plots: bool,
) -> None:
    """Execute the disruption / anomaly analysis stage.

    Parameters
    ----------
    config:
        Application configuration object (``AppConfig``).
    show_plots:
        Whether to display interactive matplotlib windows.
    """
    from predictive_analytics.disruption.analyzer import analyze_disruptions  # noqa: C0415

    console.rule("[bold red]Stage 5: Disruption Analysis")
    typer.echo("Analysing disruptions ...")
    analyze_disruptions(config=config, plot=show_plots)  # type: ignore[arg-type]
    typer.echo("Disruption analysis complete.")


@app.command()
def run(
    symbol: str = typer.Option(
        "MSFT",
        "--symbol",
        "-s",
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
        max=365,
        help="Number of time steps to forecast (1-365).",
    ),
) -> None:
    """Run the full predictive-analytics pipeline for a given stock symbol."""
    from predictive_analytics.config.logging import setup_logging  # noqa: C0415
    from predictive_analytics.config.settings import get_config  # noqa: C0415

    setup_logging()
    logger.info("Starting predictive-analytics pipeline for %s", symbol)

    try:
        config = get_config(env_file)
        output_path = _resolve_output_dir(output_dir, symbol)
        config.output_dir = str(output_path)

        if not skip_collection:
            _run_collection(config, symbol)

        if not skip_eda:
            _run_eda(config, output_path, show_plots)

        if not skip_training:
            _run_training(config, output_path)

        if not skip_forecasting:
            _run_forecasting(config, forecast_steps, show_plots)

        if not skip_disruptions:
            _run_disruptions(config, show_plots)

        console.rule("[bold green]Pipeline Complete")
        typer.echo(f"All results saved to: {output_path}")
    except KeyboardInterrupt:
        typer.echo("\nPipeline interrupted by user.", err=True)
        sys.exit(130)
    except Exception as exc:  # noqa: W0718
        logger.exception("Pipeline failed")
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
