import time
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np


class Timer:
    """
    A class to measure the execution time of functions.

    Methods
    -------
    chrono_jit(fun, *args)
        Time a JIT-compiled function and store the time.
    chrono_fun(fun, *args)
        Time a function and append the time to the list of times.
    print_to_csv(filename, rank, function_name, precision, x, y, z, px, py, backend, nodes)
        Print the timing results to a CSV file.
    """

    def __init__(self):
        self.jit_time = None
        self.times = []

    def chrono_jit(self, fun: Callable, *args) -> np.ndarray:
        """
        Time a JIT-compiled function and store the time.

        Parameters
        ----------
        fun : Callable
            The JIT-compiled function to time.
        *args :
            Arguments to pass to the function.

        Returns
        -------
        np.ndarray
            Output of the function.

        Examples
        --------
        >>> from jax import jit, random
        >>> import jax.numpy as jnp
        >>> key = random.PRNGKey(0)
        >>> x = random.normal(key, (1000, 1000))
        >>> @jit
        ... def do_fft(x):
        ...     return jnp.fft.fft(x)
        >>> timer = Timer()
        >>> x = timer.chrono_jit(do_fft, x)
        """
        start = time.perf_counter()
        out = fun(*args).block_until_ready()
        end = time.perf_counter()
        self.jit_time = (end - start) * 1e3
        return out

    def chrono_fun(self, fun: Callable, *args) -> np.ndarray:
        """
        Time a function and append the time to the list of times.

        Parameters
        ----------
        fun : Callable
            The function to time.
        *args :
            Arguments to pass to the function.

        Returns
        -------
        np.ndarray
            Output of the function.

        Examples
        --------
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>> key = random.PRNGKey(0)
        >>> x = random.normal(key, (1000, 1000))
        >>> def do_fft(x):
        ...     return jnp.fft.fft(x)
        >>> timer = Timer()
        >>> x = timer.chrono_fun(do_fft, x)
        """
        start = time.perf_counter()
        out = fun(*args).block_until_ready()
        end = time.perf_counter()
        self.times.append((end - start) * 1e3)
        return out

    def print_to_csv(self, filename: str, rank: int, function_name: str,
                     precision: str, x: int, y: int, z: int, px: int, py: int,
                     backend: str, nodes: int):
        """
        Print the timing results to a CSV file.

        Parameters
        ----------
        filename : str
            The file to write the results to.
        rank : int
            Rank of the process.
        function_name : str
            Name of the function.
        precision : str
            Precision of the operation.
        x : int
            Size of the global array in x dimension.
        y : int
            Size of the global array in y dimension.
        z : int
            Size of the global array in z dimension.
        px : int
            Processor dimensions in x direction.
        py : int
            Processor dimensions in y direction.
        backend : str
            Backend used for the operation.
        nodes : int
            Number of nodes used.

        Examples
        --------
        >>> from jax import random, jit
        >>> import jax.numpy as jnp
        >>> key = random.PRNGKey(0)
        >>> x = random.normal(key, (1000, 1000))
        >>> @jit
        ... def do_fft(x):
        ...     return jnp.fft.fft(x)
        >>> timer = Timer()
        >>> x = timer.chrono_jit(do_fft, x)
        >>> x = timer.chrono_fun(do_fft, x)
        >>> timer.print_to_csv('timing_results.csv', rank=0, function_name='FFT', precision='float32', x=1000, y=1000, z=1, px=1, py=1, backend='JAX', nodes=1)
        """
        times_array = np.array(self.times)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        last_time = times_array[-1]

        with open(filename, 'a') as f:
            f.write(
                f"{rank},{function_name},{precision},{x},{y},{z},{px},{py},{backend},{nodes},{self.jit_time:.4f},{min_time:.4f},{max_time:.4f},{mean_time:.4f},{std_time:.4f},{last_time:.4f}\n"
            )
