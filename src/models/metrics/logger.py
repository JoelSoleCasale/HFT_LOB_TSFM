"""
WandB logging utilities for metrics and plots.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from models.config import LoggingConfig


class WandbLogger:
    """WandB-based metrics and visualization logger."""

    def __init__(
        self,
        log_config: LoggingConfig,
        train_config: dict | None = None,
    ) -> None:
        """
        Initialize WandB logger.

        Args:
            log_config: Logging configuration
            train_config: Training configuration to log
        """
        self.log_config = log_config
        self.train_config = train_config or {}
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure WandB is initialized."""
        if not self._initialized:
            wandb.init(
                project=self.log_config.project_name,
                entity=self.log_config.wandb_entity,
                tags=self.log_config.wandb_tags,
                name=self.log_config.experiment_name,
                config=self.train_config,
                reinit=True,
            )
            self._initialized = True

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log numerical metrics to WandB.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number
        """
        self._ensure_initialized()
        if step is not None:
            metrics = {**metrics, "step": step}
        wandb.log(metrics)

    def log_image(self, name: str, image: object, step: int | None = None) -> None:
        """
        Log an image to WandB.

        Args:
            name: Name for the image
            image: Image object (matplotlib figure, PIL image, etc.)
            step: Optional step/epoch number
        """
        self._ensure_initialized()
        wandb.log({name: wandb.Image(image)}, step=step)

    def log_histogram(self, name: str, values: np.ndarray, step: int | None = None) -> None:
        """
        Log a histogram to WandB.

        Args:
            name: Name for the histogram
            values: Array of values to histogram
            step: Optional step/epoch number
        """
        self._ensure_initialized()
        wandb.log({name: wandb.Histogram(values)}, step=step)

    def log_plots(self, plots: dict[str, plt.Figure], step: int | None = None) -> None:
        """
        Log multiple plots to WandB.

        Args:
            plots: Dictionary mapping plot names to matplotlib figures
            step: Optional step/epoch number
        """
        for name, fig in plots.items():
            self.log_image(name, fig, step=step)
            plt.close(fig)  # Clean up

    def close(self) -> None:
        """Close WandB run."""
        if self._initialized:
            wandb.finish()
            self._initialized = False
