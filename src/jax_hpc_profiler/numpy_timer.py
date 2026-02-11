import time
from typing import Callable

import numpy as np

from .timer import AbstractTimer


class NumpyTimer(AbstractTimer):
    def chrono_jit(self, fun: Callable, *args, **kwargs):
        start = time.perf_counter()
        out = fun(*args, **kwargs)
        end = time.perf_counter()
        self.jit_time = (end - start) * 1e3
        return out

    def chrono_fun(self, fun: Callable, *args, **kwargs):
        start = time.perf_counter()
        out = fun(*args, **kwargs)
        end = time.perf_counter()
        self.times.append((end - start) * 1e3)
        return out

    def _get_mean_times(self) -> np.ndarray:
        return np.array(self.times)
