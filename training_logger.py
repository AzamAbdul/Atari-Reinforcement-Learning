"""
Training logger for managing CSV output of training metrics.

Handles logging of:
- Loss metrics per epoch
- Reward metrics per epoch
- Timing metrics per step (optional)
"""

import csv
from typing import List, Optional


class TrainingLogger:
    """Handles CSV logging for training metrics."""

    def __init__(self, timestamp: str, log_timing: bool = False):
        """
        Initialize training loggers.

        Args:
            timestamp: Timestamp string for log filenames
            log_timing: Whether to enable detailed timing logs
        """
        self.timestamp = timestamp
        self.log_timing = log_timing

        # Initialize loss logger
        self.loss_logger = self._create_csv_logger(
            filename=f'training_loss_{timestamp}.csv',
            headers=['epoch', 'avg_loss']
        )

        # Initialize rewards logger
        self.rewards_logger = self._create_csv_logger(
            filename=f'training_rewards_{timestamp}.csv',
            headers=['epoch', 'total_reward']
        )

        # Initialize timing logger if enabled
        self.timing_logger: Optional[str] = None
        self.timing_buffer: List[List] = []
        if log_timing:
            self.timing_logger = self._create_csv_logger(
                filename=f'training_timing_{timestamp}.csv',
                headers=['step', 'epoch', 'action_selection_ms', 'env_step_ms',
                        'preprocessing_ms', 'memory_add_ms', 'online_net_training_ms',
                        'target_net_training_ms', 'total_step_ms']
            )
            print(f"Timing data will be logged to training_timing_{timestamp}.csv every 10000 steps")
        else:
            print("Timing logging disabled - use --log-timing to enable")

    def _create_csv_logger(self, filename: str, headers: List[str]) -> str:
        """
        Creates CSV file with headers and returns filename.

        Args:
            filename: Name of CSV file to create
            headers: List of column headers

        Returns:
            filename: The created filename
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        print(f"{headers[1].replace('_', ' ').title()} data will be logged to {filename} after each epoch")
        return filename

    def log_loss(self, epoch: int, avg_loss: float):
        """
        Append loss data to CSV.

        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
        """
        with open(self.loss_logger, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss])

    def log_reward(self, epoch: int, total_reward: float):
        """
        Append reward data to CSV.

        Args:
            epoch: Current epoch number
            total_reward: Total reward accumulated in the epoch
        """
        with open(self.rewards_logger, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, total_reward])

    def add_timing_data(self, timing_data: List):
        """
        Buffer timing data for batch writing.

        Args:
            timing_data: List containing timing measurements
        """
        if not self.log_timing:
            return

        self.timing_buffer.append(timing_data)

    def flush_timing_buffer(self):
        """Write buffered timing data to CSV and clear buffer."""
        if self.timing_buffer and self.timing_logger:
            with open(self.timing_logger, 'a', newline='') as f:
                csv.writer(f).writerows(self.timing_buffer)
            print(f"Wrote {len(self.timing_buffer)} timing records to {self.timing_logger}")
            self.timing_buffer.clear()
