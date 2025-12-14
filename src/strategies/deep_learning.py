"""
Deep learning strategies that convert trained models into trading strategies.

These strategies bridge the gap between PyTorch models and hftbacktest's
numba-compiled trading loops.
"""

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import torch
from numba import njit
from hftbacktest import LIMIT, MARKET, GTC

from .base import BaseStrategy, StrategyConfig
from models.architectures.base import FinancialTimeSeriesModel

from hftbacktest import HashMapMarketDepthBacktest_TypeHint


@dataclass
class DeepLearningStrategyConfig(StrategyConfig):
    """Configuration for deep learning strategies."""

    sequence_length: int = 128  # Lookback window for features


class DeepLearningStrategy(BaseStrategy):
    """
    Base class for strategies that use deep learning models.

    This class handles the conversion of a FinancialTimeSeriesModel into
    a strategy that can be executed with hftbacktest. It manages:
    - Feature preprocessing and batching
    - Model inference
    - Conversion of model outputs to trading signals

    Subclasses should implement `generate_signals` to convert model predictions
    into trading actions.
    """

    def __init__(self, model: FinancialTimeSeriesModel, config: DeepLearningStrategyConfig):
        """
        Initialize the deep learning strategy.

        Args:
            model: Trained PyTorch model
            config: Strategy configuration
        """
        super().__init__(config)
        self.model = model
        self.model.eval()  # Set to evaluation mode
        self.device = next(model.parameters()).device
        self.config: DeepLearningStrategyConfig = config

    def prepare_data(
        self, features: np.ndarray, timestamps: np.ndarray = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Prepare features for inference during backtesting.

        Converts features into model predictions that can be used by
        the numba-compiled trading loop.

        Args:
            features: Feature array of shape (n_samples, n_features)
            timestamps: Timestamp array (optional)
            **kwargs: Additional data

        Returns:
            dict: Dictionary containing predictions and timestamps
        """
        n_samples = features.shape[0]
        seq_len = self.config.sequence_length

        if n_samples < seq_len:
            raise ValueError(f"Not enough samples ({n_samples}) for sequence length ({seq_len})")

        # Generate predictions for all valid sequences
        # We can make predictions starting from index seq_len-1 (first complete sequence)
        # up to index n_samples-1 (last sample)
        num_predictions = n_samples - seq_len + 1
        predictions = []

        with torch.no_grad():
            # Process in batches for efficiency
            batch_size = 1024
            for batch_start in range(0, num_predictions, batch_size):
                batch_end = min(batch_start + batch_size, num_predictions)

                # Create sequences for this batch
                # Each sequence ends at index: seq_len - 1 + batch_start + offset
                sequences = []
                for offset in range(batch_end - batch_start):
                    end_idx = seq_len - 1 + batch_start + offset
                    start_idx = end_idx - seq_len + 1
                    seq = features[start_idx : end_idx + 1]
                    sequences.append(seq)

                sequences = np.stack(sequences)
                sequences_tensor = torch.FloatTensor(sequences).to(self.device)

                # Get predictions
                outputs = self.model(sequences_tensor)
                predictions.append(outputs.cpu().numpy())

        predictions = np.vstack(predictions)

        # Align timestamps with predictions (skip initial sequence_length - 1 samples)
        if timestamps is not None:
            aligned_timestamps = timestamps[seq_len - 1 :]
        else:
            aligned_timestamps = np.arange(seq_len - 1, n_samples)

        return {
            "predictions": predictions,
            "timestamps": aligned_timestamps,
            "features": features,
        }

    def run(self, hbt: HashMapMarketDepthBacktest_TypeHint) -> bool:
        """
        This method should be overridden by subclasses.

        The actual trading loop must be a standalone numba-compiled function
        that takes preprocessed predictions as input.
        """
        raise NotImplementedError(
            "DeepLearningStrategy.run() must be implemented by subclasses. "
            "Use create_strategy_function() to generate a numba-compatible function."
        )


@dataclass
class ClassificationStrategyConfig(DeepLearningStrategyConfig):
    """Configuration for classification-based trading strategies."""

    buy_threshold: float = 0.6  # Probability threshold for buy signal (class 1)
    sell_threshold: float = 0.6  # Probability threshold for sell signal (class -1)
    order_quantity: float = 1.0  # Quantity to trade
    max_position: float = 10.0  # Maximum absolute position size
    use_market_orders: bool = False  # Use market orders instead of limit orders
    limit_offset_ticks: int = 0  # Offset for limit orders (0 = at best bid/ask)


class ClassificationStrategy(DeepLearningStrategy):
    """
    Strategy for classification models that predict directional moves.

    This strategy converts model predictions into buy/sell signals based on
    probability thresholds. The model should output logits or probabilities
    for three classes: [-1, 0, 1] representing sell, hold, buy.

    The strategy:
    1. Converts logits to probabilities via softmax
    2. Checks if max probability exceeds threshold
    3. Executes trade if threshold is met and position limits allow
    """

    def __init__(self, model: FinancialTimeSeriesModel, config: ClassificationStrategyConfig):
        """
        Initialize the classification strategy.

        Args:
            model: Trained classification model (output_size should be 3)
            config: Strategy configuration
        """
        super().__init__(model, config)
        self.config: ClassificationStrategyConfig = config

        # Validate model output size
        if model.output_size != 3:
            raise ValueError(
                f"Classification model must have output_size=3, got {model.output_size}"
            )

        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        super().validate_config()

        if not 0 < self.config.buy_threshold <= 1:
            raise ValueError("buy_threshold must be in (0, 1]")
        if not 0 < self.config.sell_threshold <= 1:
            raise ValueError("sell_threshold must be in (0, 1]")
        if self.config.order_quantity <= 0:
            raise ValueError("order_quantity must be positive")
        if self.config.max_position <= 0:
            raise ValueError("max_position must be positive")

    def prepare_data(
        self, features: np.ndarray, timestamps: np.ndarray = None, **kwargs
    ) -> dict[str, np.ndarray]:
        """
        Prepare data and convert logits to probabilities.

        Returns predictions as probabilities for classes [-1, 0, 1].
        """
        data = super().prepare_data(features, timestamps, **kwargs)

        # Convert logits to probabilities via softmax
        logits = data["predictions"]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        data["probabilities"] = probabilities
        return data

    def create_strategy_function(
        self, predictions: np.ndarray, timestamps: np.ndarray
    ) -> Callable:
        """
        Create a numba-compiled strategy function.

        This function can be passed to hftbacktest for execution.

        Args:
            predictions: Probability predictions (n_samples, 3)
            timestamps: Corresponding timestamps
            use_recorder: Whether to use recorder for statistics (default: True)

        Returns:
            Callable: Numba-compiled strategy function that accepts (hbt, recorder)

        Example:
            >>> strategy = ClassificationStrategy(model, config)
            >>> data = strategy.prepare_data(features, timestamps)
            >>> strategy_func = strategy.create_strategy_function(
            ...     data["probabilities"], data["timestamps"]
            ... )
            >>> hbt = HashMapMarketDepthBacktest([asset])
            >>> recorder = Recorder(hbt.num_assets, 1_000_000)
            >>> strategy_func(hbt, recorder.recorder)
            >>> hbt.close()
        """
        buy_threshold = self.config.buy_threshold
        sell_threshold = self.config.sell_threshold
        order_quantity = self.config.order_quantity
        max_position = self.config.max_position
        asset_no = self.config.asset_no
        use_market_orders = self.config.use_market_orders
        order_type = MARKET if self.config.use_market_orders else LIMIT
        limit_offset_ticks = self.config.limit_offset_ticks

        @njit
        def strategy_function(hbt: HashMapMarketDepthBacktest_TypeHint, recorder=None):
            order_id_counter = 1

            # Initialize the backtest by elapsing to the first event
            # Before this, hbt.current_timestamp returns max int64
            if hbt.elapse(0) != 0:
                # No data in backtest
                return

            # Now get the actual backtest start time
            start_time = hbt.current_timestamp

            # Find first prediction timestamp >= backtest start time
            prediction_idx = 0
            while prediction_idx < len(timestamps) and timestamps[prediction_idx] < start_time:
                prediction_idx += 1

            # Check if all predictions are before the backtest start
            if prediction_idx >= len(timestamps):
                print(
                    f"Warning: All model predictions ({len(timestamps)}) are before "
                    f"backtest start time {start_time}. No trades will be executed."
                )
                return

            # Main trading loop - process each prediction timestamp
            while prediction_idx < len(timestamps):
                pred_time = timestamps[prediction_idx]

                # Elapse time to the prediction timestamp
                time_to_elapse = pred_time - hbt.current_timestamp
                if time_to_elapse > 0:
                    if hbt.elapse(time_to_elapse) != 0:
                        break  # Backtest ended

                # Clear filled/canceled orders
                hbt.clear_inactive_orders(asset_no)

                # Record state before processing prediction (if recorder is provided)
                if recorder is not None:
                    recorder.record(hbt)

                # Get prediction probabilities [P(-1), P(0), P(1)]
                probs = predictions[prediction_idx]

                action = probs.argmax() - 1  # -1, 0, or 1
                action_prob = probs[action + 1]

                # Determine signal
                signal = 0
                if action == 1 and action_prob >= buy_threshold:
                    signal = 1
                elif action == -1 and action_prob >= sell_threshold:
                    signal = -1

                # Get current position and market depth
                current_position = hbt.position(asset_no)
                depth = hbt.depth(asset_no)

                # Execute trades based on signal and position limits
                if signal == 1 and current_position < max_position:
                    # Buy signal
                    quantity = min(order_quantity, max_position - current_position)

                    order_price = depth.best_ask
                    if not use_market_orders:
                        order_price += limit_offset_ticks * depth.tick_size

                    hbt.submit_buy_order(
                        asset_no,
                        order_id_counter,
                        order_price,
                        quantity,
                        GTC,
                        order_type,
                        False,
                    )

                    order_id_counter += 1

                elif signal == -1 and current_position > -max_position:
                    # Sell signal
                    quantity = min(order_quantity, max_position + current_position)

                    order_price = depth.best_bid
                    if not use_market_orders:
                        order_price -= limit_offset_ticks * depth.tick_size

                    hbt.submit_sell_order(
                        asset_no,
                        order_id_counter,
                        order_price,
                        quantity,
                        GTC,
                        order_type,
                        False,
                    )

                    order_id_counter += 1

                prediction_idx += 1

            return

        return strategy_function

    def run(self, hbt: Any) -> bool:
        """
        Not directly callable. Use create_strategy_function() instead.

        This method exists to satisfy the abstract base class but should not
        be called directly. Instead, use:

        1. strategy.prepare_data() to get predictions
        2. strategy.create_strategy_function() to get numba function
        3. Pass that function to hftbacktest
        """
        raise NotImplementedError(
            "ClassificationStrategy.run() should not be called directly. "
            "Use create_strategy_function() to generate the trading function."
        )
