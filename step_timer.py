import time
from typing import Dict


class StepTimer:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, float] = {}

    def time(self, name: str):
        return self._TimingContext(self, name)

    def start(self, name: str):
        if self.enabled:
            self.timings[f'_start_{name}'] = time.time()

    def stop(self, name: str):
        if self.enabled:
            start_key = f'_start_{name}'
            if start_key in self.timings:
                elapsed = (time.time() - self.timings[start_key]) * 1000
                self.timings[name] = elapsed
                del self.timings[start_key]

    def get(self, name: str, default: float = 0.0) -> float:
        return self.timings.get(name, default)

    def reset(self):
        self.timings.clear()

    class _TimingContext:
        def __init__(self, timer: 'StepTimer', name: str):
            self.timer = timer
            self.name = name

        def __enter__(self):
            self.timer.start(self.name)
            return self

        def __exit__(self, *args):
            self.timer.stop(self.name)
