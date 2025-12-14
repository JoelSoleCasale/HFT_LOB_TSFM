"""
Example of using the classification strategy with a random model.

This script demonstrates:
1. Creating a simple random model that outputs random logits
2. Preparing features for backtesting
3. Creating a strategy from the model
4. Running the strategy with hftbacktest using raw incremental orderbook data
5. Analyzing the results
"""

import polars as pl
from datetime import date

from definitions import ROOT_DIR
from core.orderbook import OrderBook
from features import FeaturePipeline, FeatureExtractorRegistry, InputSpace
from strategies import ClassificationStrategy, ClassificationStrategyConfig
from utils import date_range, get_hftbacktest_array
from data_manager.downloader.data_downloader import DataDownloader
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, Recorder
import torch


class RandomModel:
    """A dummy model that returns random predictions."""

    def __init__(self, input_size: int, output_size: int = 3):
        self.input_size = input_size
        self.output_size = output_size
        # Create a dummy parameter to satisfy device detection
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        """Return iterator of parameters (needed for device detection)."""
        return iter([self._dummy_param])

    def eval(self):
        """Set model to eval mode (no-op for random model)."""
        return self

    def __call__(self, x):
        """Generate random predictions for input batch."""
        batch_size = x.shape[0]
        device = x.device
        # Return random logits on the same device as input
        return torch.randn(batch_size, self.output_size, device=device)


def main():
    # -------------------------------------------------------------------------
    # 1. Load orderbook data
    # -------------------------------------------------------------------------
    print("Loading orderbook data...")
    ob_paths = [
        ROOT_DIR
        / f'data/orderbook_snapshots/binance_futures/BTCUSDT/{d.strftime("%Y-%m-%d")}_L20.parquet'
        for d in date_range(date(2025, 7, 1), date(2025, 7, 3))
    ]

    orderbook = (
        OrderBook.from_parquet(ob_paths, lazy=True)
        .select_levels(5)
        .sample_by_time(1_000_000_000)  # 1 second sampling
    )

    # -------------------------------------------------------------------------
    # 2. Extract features
    # -------------------------------------------------------------------------
    print("Extracting features...")
    input_space = InputSpace(orderbook_snapshots=orderbook)

    feature_pipeline = (
        FeaturePipeline()
        .add_extractor(FeatureExtractorRegistry.create("mid_price"))
        .add_extractor(FeatureExtractorRegistry.create("spread"))
        .add_extractor(FeatureExtractorRegistry.create("orderbook_imbalance"))
    )

    features_df = feature_pipeline.extract_all(input_space).collect()
    print(f"Features shape: {features_df.shape}")

    # Extract features as numpy array (excluding timestamp)
    feature_cols = [col for col in features_df.columns if col != "timestamp"]
    features = features_df.select(feature_cols).to_numpy()
    timestamps = features_df["timestamp"].to_numpy()

    # -------------------------------------------------------------------------
    # 3. Create random model
    # -------------------------------------------------------------------------
    print("Creating random model...")
    # Create a simple random model that returns random logits
    model = RandomModel(
        input_size=len(feature_cols),
        output_size=3,  # 3 classes: sell (-1), hold (0), buy (1)
    )

    # Set to evaluation mode (no training)
    model.eval()
    print(f"Created random model with {len(feature_cols)} input features")

    # -------------------------------------------------------------------------
    # 4. Create strategy
    # -------------------------------------------------------------------------
    print("Creating strategy...")
    strategy_config = ClassificationStrategyConfig(
        sequence_length=128,
        buy_threshold=0.6,  # Low threshold for demo with random model
        sell_threshold=0.6,  # Low threshold for demo with random model
        order_quantity=0.01,  # Small quantity for crypto
        max_position=1,
        use_market_orders=True,
        limit_offset_ticks=0,
    )

    strategy = ClassificationStrategy(model, strategy_config)

    # Prepare predictions
    print("Generating predictions (from random model - totally random outputs)...")
    prepared_data = strategy.prepare_data(features, timestamps)
    probabilities = prepared_data["probabilities"]
    pred_timestamps = prepared_data["timestamps"]

    print(f"Generated {len(probabilities)} predictions")
    print(f"Time range: {pred_timestamps[0]} to {pred_timestamps[-1]}")

    # Show some prediction statistics
    print("\nPrediction statistics:")
    print(f"  Mean buy probability: {probabilities[:, 2].mean():.3f}")
    print(f"  Mean sell probability: {probabilities[:, 0].mean():.3f}")
    print(f"  Mean hold probability: {probabilities[:, 1].mean():.3f}")
    print(f"  Max buy probability: {probabilities[:, 2].max():.3f}")
    print(f"  Max sell probability: {probabilities[:, 0].max():.3f}")
    print(
        f"  Predictions with buy signal (>={strategy_config.buy_threshold}): {(probabilities[:, 2] >= strategy_config.buy_threshold).sum()}"
    )
    print(
        f"  Predictions with sell signal (>={strategy_config.sell_threshold}): {(probabilities[:, 0] >= strategy_config.sell_threshold).sum()}"
    )
    print("\nNote: This is a random model that generates completely random predictions.")

    # -------------------------------------------------------------------------
    # 5. Prepare hftbacktest data from raw incremental orderbook
    # -------------------------------------------------------------------------
    print("Preparing backtest data from raw incremental orderbook...")

    # Initialize DataDownloader to get raw incremental orderbook data
    downloader = DataDownloader()

    # Download raw orderbook data for the same date range
    # This is the incremental orderbook format required by hftbacktest
    print("Loading raw incremental orderbook data...")
    raw_orderbook_dfs = []
    for day in date_range(date(2025, 7, 1), date(2025, 7, 1)):
        try:
            # Get raw orderbook data (incremental format)
            raw_df = downloader.get_data(
                data_type="orderbook",
                symbol="BTCUSDT",
                exchange="binance_futures",
                date=day,
                reference_ts="received_time",
            )
            raw_orderbook_dfs.append(raw_df)
            print(f"Loaded data for {day}")
        except Exception as e:
            print(f"Warning: Could not load data for {day}: {e}")
            print("Continuing with available data...")

    if not raw_orderbook_dfs:
        raise RuntimeError("No raw orderbook data could be loaded. Please download data first.")

    # Concatenate all days and collect
    raw_orderbook_df = pl.concat(raw_orderbook_dfs).collect()
    print(f"Raw orderbook data shape: {raw_orderbook_df.shape}")

    # Convert to hftbacktest format
    hft_data = get_hftbacktest_array(raw_orderbook_df)
    print(f"Converted to hftbacktest array with {len(hft_data)} events")

    # Create backtest asset
    asset = (
        BacktestAsset()
        .data([hft_data])
        .linear_asset(1.0)
        .constant_order_latency(10_000_000, 10_000_000)  # 10ms latency (order and response)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0002, 0.0007)  # 0.02% maker, 0.07% taker
        .tick_size(0.1)
        .lot_size(0.001)
    )

    # -------------------------------------------------------------------------
    # 6. Run backtest
    # -------------------------------------------------------------------------
    print("Running backtest...")
    hbt = HashMapMarketDepthBacktest([asset])

    # Create recorder for statistics (capacity should match or exceed data size)
    recorder = Recorder(hbt.num_assets, 100_000_000)  # 100M capacity for 78M events

    # Create and run strategy function with recorder
    strategy_func = strategy.create_strategy_function(probabilities, pred_timestamps)
    strategy_func(hbt, recorder.recorder)

    print("Backtest complete!")

    # -------------------------------------------------------------------------
    # 7. Analyze results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    state = hbt.state_values(0)
    position = hbt.position(0)

    print(f"\nFinal Position: {position}")
    print(f"Final Balance: ${state.balance:.2f}")
    print(f"Total Fees: ${state.fee:.2f}")
    print(f"Net P&L: ${state.balance - state.fee:.2f}")
    print(f"Total Trades: {state.num_trades}")

    # Close backtest
    hbt.close()

    # -------------------------------------------------------------------------
    # 8. Display recorder statistics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERFORMANCE STATISTICS")
    print("=" * 80)

    # Import after closing backtest
    from hftbacktest.stats import LinearAssetRecord

    # Create record from recorder data
    record = LinearAssetRecord(recorder.get(0))

    print("\nRaw Recorder Data:")
    print(recorder.get(0))

    # Generate statistics
    stats = record.stats()

    # Display summary
    print(stats.summary(pretty=True))

    # Plot statistics (will open in default viewer)
    print("\nGenerating performance plots...")
    fig = stats.plot()

    # save the plot
    fig.savefig(ROOT_DIR / "examples" / "strategy_example_performance.png")

    print("\nBacktest finished successfully!")


if __name__ == "__main__":
    main()
