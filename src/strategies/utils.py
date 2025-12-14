"""
Utility functions for running backtests and analyzing strategy performance.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
from hftbacktest import HashMapMarketDepthBacktest, BacktestAsset, Recorder
from hftbacktest.stats import LinearAssetRecord, Stats
import matplotlib.pyplot as plt
from loguru import logger

from models.architectures.base import FinancialTimeSeriesModel
from strategies.deep_learning import ClassificationStrategy, ClassificationStrategyConfig
from utils import get_hftbacktest_array


def backtest_model(
    model: FinancialTimeSeriesModel,
    features: pl.LazyFrame,
    raw_orderbook_df: pl.DataFrame,
    config: Optional[ClassificationStrategyConfig] = None,
    save_plot_path: Optional[Path] = None,
) -> Optional[Stats]:
    """
    Run a backtest for a trained financial time series model and generate performance statistics.

    This function:
    1. Creates a classification strategy from the model
    2. Generates predictions on the features
    3. Converts raw orderbook data to hftbacktest format
    4. Runs the backtest with a recorder
    5. Generates and optionally saves performance statistics

    Parameters
    ----------
    model : FinancialTimeSeriesModel
        Trained model to backtest
    features : pl.LazyFrame
        Feature data with timestamp column (same as used for training)
    raw_orderbook_df : pl.DataFrame
        Raw incremental orderbook data (from DataDownloader.get_data())
        Must have columns: timestamp, is_trade, price, qty, side, ev
    config : ClassificationStrategyConfig, optional
        Strategy configuration. If None, uses default thresholds (0.6 for buy/sell)
    save_plot_path : Path, optional
        If provided, saves the performance plot to this path (e.g., "results/backtest.png")

    Returns
    -------
    Optional[Stats]
        hftbacktest Stats object with performance metrics, or None if stats could not be computed

    Returns
    -------
    stats : Stats
        The hftbacktest Stats object containing all performance metrics and methods
        for generating summaries and plots. Access metrics via:
        - stats.summary(): Returns a polars DataFrame with all metrics
        - stats.plot(): Generates performance plots
        - Individual metric access through the Stats object

    Example
    -------
    >>> from models.architectures.base import FinancialTimeSeriesModel
    >>> from strategies.utils import backtest_model
    >>> from strategies import ClassificationStrategyConfig
    >>> from data_manager.downloader import DataDownloader
    >>>
    >>> # Load trained model
    >>> model = FinancialTimeSeriesModel.load("model_checkpoint/trained_model.pth")
    >>>
    >>> # Get raw orderbook data
    >>> downloader = DataDownloader(...)
    >>> raw_data = downloader.get_data(...)
    >>>
    >>> # Run backtest
    >>> config = ClassificationStrategyConfig(buy_threshold=0.7, sell_threshold=0.7)
    >>> stats = backtest_model(
    ...     model, features, raw_data, config,
    ...     save_plot_path=Path("results/backtest.png")
    ... )
    >>> # Access metrics
    >>> summary_df = stats.summary()
    >>> print(f"Sharpe Ratio: {summary_df['SR'][0]:.2f}")
    >>> stats.plot()  # Display plots
    """
    # Use default config if none provided
    if config is None:
        config = ClassificationStrategyConfig()

    logger.info("Creating classification strategy from model")
    strategy = ClassificationStrategy(model, config)

    logger.info("Preparing feature data")
    # Convert LazyFrame to DataFrame if needed
    if hasattr(features, "collect"):
        features_df = features.collect()
    else:
        features_df = features

    # Extract timestamps and convert features to NumPy
    timestamps = features_df["timestamp"].to_numpy()
    feature_columns = [col for col in features_df.columns if col != "timestamp"]
    features_array = features_df.select(feature_columns).to_numpy()

    logger.info("Generating predictions from model")
    prepared_data = strategy.prepare_data(features_array, timestamps)
    probabilities = prepared_data["probabilities"]
    pred_timestamps = prepared_data["timestamps"]
    logger.info(f"Generated {len(probabilities)} predictions")
    logger.debug(f"Prediction time range: {pred_timestamps[0]} to {pred_timestamps[-1]}")
    logger.debug(f"Predicted probabilities sample (first 5): {probabilities[:5]}")
    logger.debug(f"Predicted probabilities sample (last 5): {probabilities[-5:]}")

    # Check for prediction diversity - warn if model is predicting constant values
    # Skip first row which may contain NaN
    valid_probs = probabilities[~np.isnan(probabilities).any(axis=1)]  # Remove any rows with NaN
    prob_variance = valid_probs.var(axis=0)
    logger.info(f"Prediction variance per class: {prob_variance}")

    # Check if variance is very low (threshold of 1e-4 is reasonable for probabilities)
    if len(valid_probs) > 0 and (prob_variance < 1e-4).all():
        logger.warning("=" * 80)
        logger.warning("WARNING: Model predictions show very low variance!")
        logger.warning("=" * 80)
        logger.warning(
            "The model appears to be outputting nearly identical predictions for all inputs."
        )
        logger.warning("This typically indicates:")
        logger.warning("  1. The model is undertrained or has not learned useful patterns")
        logger.warning("  2. The model has collapsed to predicting the majority class")
        logger.warning("  3. There may be an issue with input data normalization/scaling")
        logger.warning(f"Prediction variance per class: {prob_variance}")
        logger.warning("Consider retraining the model with:")
        logger.warning("  - More epochs")
        logger.warning("  - Better class balancing (e.g., weighted loss)")
        logger.warning("  - Different model architecture")
        logger.warning("  - Feature normalization/standardization")
        logger.warning("=" * 80)

    logger.info("Converting orderbook data to hftbacktest format")
    hft_data = get_hftbacktest_array(raw_orderbook_df)
    logger.info(f"Converted to hftbacktest array with {len(hft_data)} events")
    logger.debug(f"Orderbook time range: {hft_data['exch_ts'][0]} to {hft_data['exch_ts'][-1]}")

    # Create backtest asset
    logger.info("Setting up backtest asset with latency and fee models")
    asset = (
        BacktestAsset()
        .data([hft_data])
        .linear_asset(1.0)
        .constant_order_latency(10_000_000, 10_000_000)  # 10ms latency
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0002, 0.0007)  # 0.02% maker, 0.07% taker
        .tick_size(0.1)
        .lot_size(0.001)
    )

    # Create backtest
    logger.info("Running backtest")
    hbt = HashMapMarketDepthBacktest([asset])

    # Create recorder with sufficient capacity
    recorder_capacity = max(len(probabilities) * 2, 1_000_000)
    recorder = Recorder(hbt.num_assets, recorder_capacity)

    # Create and run strategy function with recorder
    strategy_func = strategy.create_strategy_function(probabilities, pred_timestamps)
    strategy_func(hbt, recorder.recorder)

    # Get basic results before closing
    state = hbt.state_values(0)
    position = hbt.position(0)

    logger.info("Backtest complete")
    logger.info(f"Final Position: {position}")
    logger.info(f"Final Balance: ${state.balance:.2f}")
    logger.info(f"Total Fees: ${state.fee:.2f}")
    logger.info(f"Net P&L: ${state.balance - state.fee:.2f}")
    logger.info(f"Total Trades: {state.num_trades}")

    # Close backtest
    hbt.close()

    # Generate statistics
    logger.info("Generating performance statistics")
    record = LinearAssetRecord(recorder.get(0))

    try:
        stats = record.stats()
    except (TypeError, ZeroDivisionError, ValueError) as e:
        logger.warning(f"Could not compute some statistics: {e}")
        logger.warning("This may be due to insufficient data or no variance in returns")
        # Try with a minimal set of metrics to avoid computation errors
        from hftbacktest.stats.metrics import MaxDrawdown, NumberOfTrades, Ret

        try:
            stats = record.stats(metrics=[Ret(), MaxDrawdown(), NumberOfTrades()])
        except Exception as e2:
            logger.error(f"Failed to compute even basic statistics: {e2}")
            logger.error("Returning None for stats. Check backtest configuration and data.")
            return None

    # Generate and optionally save plot
    if save_plot_path is not None and stats is not None:
        logger.info("Generating performance plots")
        try:
            fig = stats.plot()
            # Create directory if it doesn't exist
            save_plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Plot saved to: {save_plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate plot: {e}")

    return stats


def print_backtest_summary(stats: Optional[Stats]) -> None:
    """
    Print a formatted summary of backtest statistics.

    Parameters
    ----------
    stats : Stats or None
        hftbacktest Stats object from backtest_model(), or None if stats could not be computed
    """
    if stats is None:
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print("Statistics could not be computed (insufficient data or variance)")
        print("=" * 80)
        return

    stats_df = stats.summary()

    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    if "SR" in stats_df.columns:
        print(f"Sharpe Ratio:        {float(stats_df['SR'][0]):>12.4f}")
    if "Sortino" in stats_df.columns:
        print(f"Sortino Ratio:       {float(stats_df['Sortino'][0]):>12.4f}")
    if "MDD" in stats_df.columns:
        print(f"Max Drawdown:        {float(stats_df['MDD'][0]):>12.2%}")
    if "Return" in stats_df.columns:
        print(f"Total Return:        {float(stats_df['Return'][0]):>12.2%}")

    print("=" * 80)
