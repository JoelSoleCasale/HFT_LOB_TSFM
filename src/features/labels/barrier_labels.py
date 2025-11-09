"""Triple barrier method label extractors"""

import polars as pl
from features.base.label_extractor import BaseLabelExtractor
from features.base.input_space import InputSpace


class TripleBarrierLabel(BaseLabelExtractor):
    """Triple barrier method labels: 1 (take profit hit first), -1 (stop loss hit first), 0 (time barrier hit first)

    The triple barrier method is a labeling technique that simulates opening a position
    and exiting when one of three barriers is hit first:
    1. Upper barrier (take profit): price increases by threshold %
    2. Lower barrier (stop loss): price decreases by threshold %
    3. Time barrier (horizon): maximum holding period expires

    Args:
        threshold (float): Percentage threshold for profit/loss barriers (e.g., 0.01 for 1%)
        horizon (int): Maximum number of time steps to hold position (time barrier)

    Returns:
        Labels: 1 if upper barrier hit first, -1 if lower barrier hit first, 0 if time barrier hit first
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.01)  # 1% default threshold
        self.horizon = self.config.get("horizon", 10)  # samples
        self.dependencies = ["orderbook_snapshots"]
        self.label_names = [f"triple_barrier_{self.threshold}_{self.horizon}"]

    def extract(self, input_space: InputSpace) -> pl.LazyFrame:
        """
        Extract triple barrier labels.

        For each timestamp t, we:
        1. Get the entry price at time t
        2. Calculate upper barrier = entry_price * (1 + threshold)
        3. Calculate lower barrier = entry_price * (1 - threshold)
        4. Look forward up to horizon steps to find which barrier is hit first
        5. Assign label: 1 (upper), -1 (lower), 0 (time barrier/no hit)
        """
        self.validate_input(input_space)

        # Get mid prices as our reference price
        mid_prices = input_space.orderbook_snapshots.get_mid_prices()

        # Collect to DataFrame for vectorized operations
        # Note: This is necessary because we need to look forward with complex logic
        if input_space.orderbook_snapshots.is_lazy:
            df = mid_prices.collect()
        else:
            df = mid_prices

        # Calculate barriers for each entry point
        df = df.with_columns(
            [
                (pl.col("mid_price") * (1 + self.threshold)).alias("upper_barrier"),
                (pl.col("mid_price") * (1 - self.threshold)).alias("lower_barrier"),
            ]
        )

        # Create a function to find which barrier is hit first
        def find_first_barrier(prices, upper, lower, horizon):
            """
            For each price point, look forward to find which barrier is hit first.

            Args:
                prices: Array of prices
                upper: Array of upper barriers
                lower: Array of lower barriers
                horizon: Max steps to look forward

            Returns:
                Array of labels: 1, -1, or 0
            """
            n = len(prices)
            labels = [0] * n

            for i in range(n):
                upper_thresh = upper[i]
                lower_thresh = lower[i]

                # Look forward up to horizon steps
                max_look = min(i + horizon + 1, n)

                for j in range(i + 1, max_look):
                    future_price = prices[j]

                    # Check if upper barrier hit first
                    if future_price >= upper_thresh:
                        labels[i] = 1
                        break

                    # Check if lower barrier hit first
                    if future_price <= lower_thresh:
                        labels[i] = -1
                        break

                # If no barrier hit within horizon, label remains 0

            return labels

        # Extract arrays and compute labels
        prices = df["mid_price"].to_list()
        upper = df["upper_barrier"].to_list()
        lower = df["lower_barrier"].to_list()

        labels = find_first_barrier(prices, upper, lower, self.horizon)

        # Add labels back to dataframe
        result = df.with_columns(
            pl.Series(name=f"triple_barrier_{self.threshold}_{self.horizon}", values=labels)
        )

        # Return as LazyFrame with only timestamp and label
        return result.select(
            ["timestamp", f"triple_barrier_{self.threshold}_{self.horizon}"]
        ).lazy()
