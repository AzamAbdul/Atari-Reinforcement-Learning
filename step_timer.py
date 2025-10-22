"""
Step timer for tracking execution time of training loop operations.

Supports both context manager and method-based timing patterns.
"""

import time
from typing import Dict


class StepTimer:
    """Flexible timer supporting both context manager and method-based timing."""

    def __init__(self, enabled: bool = True):
        """
        Initialize the step timer.

        Args:
            enabled: Whether timing is enabled. If False, all operations are no-ops.
        """
        self.enabled = enabled
        self.timings: Dict[str, float] = {}

    def time(self, name: str):
        """
        Returns a context manager for timing a block.

        Args:
            name: Name of the operation being timed

        Returns:
            Context manager that times the enclosed block

        Example:
            with timer.time('action_selection'):
                action = agent.choose_action(state)
        """
        return self._TimingContext(self, name)

    def start(self, name: str):
        """
        Start a named timer.

        Args:
            name: Name of the operation being timed
        """
        if self.enabled:
            self.timings[f'_start_{name}'] = time.time()

    def stop(self, name: str):
        """
        Stop a named timer and record elapsed time in milliseconds.

        Args:
            name: Name of the operation being timed
        """
        if self.enabled:
            start_key = f'_start_{name}'
            if start_key in self.timings:
                elapsed = (time.time() - self.timings[start_key]) * 1000
                self.timings[name] = elapsed
                del self.timings[start_key]

    def get(self, name: str, default: float = 0.0) -> float:
        """
        Get timing result for a named operation.

        Args:
            name: Name of the operation
            default: Default value if timing not found

        Returns:
            Elapsed time in milliseconds, or default if not found
        """
        return self.timings.get(name, default)

    def reset(self):
        """Clear all timings."""
        self.timings.clear()

    class _TimingContext:
        """Internal context manager for timing blocks."""

        def __init__(self, timer: 'StepTimer', name: str):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer.start(self.name)
            return self

        def __exit__(self, *args):
            self.timer.stop(self.name)
