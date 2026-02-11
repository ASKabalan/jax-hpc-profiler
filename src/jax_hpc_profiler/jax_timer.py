import time
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr, shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from .timer import AbstractTimer


class JaxTimer(AbstractTimer):
    def __init__(
        self,
        save_jaxpr=False,
        compile_info=True,
        devices=None,
        static_argnums=(),
    ):
        super().__init__()
        self.save_jaxpr = save_jaxpr
        self.compile_info = compile_info
        self.devices = devices
        self.static_argnums = static_argnums

    def _should_write(self) -> bool:
        return jax.process_index() == 0

    def _has_compile_info(self) -> bool:
        return self.compile_info

    def _read_memory_analysis(self, memory_analysis: Any) -> Tuple:
        if memory_analysis is None:
            return None, None, None, None
        return (
            memory_analysis.generated_code_size_in_bytes,
            memory_analysis.argument_size_in_bytes,
            memory_analysis.output_size_in_bytes,
            memory_analysis.temp_size_in_bytes,
        )

    def chrono_jit(self, fun: Callable, *args, **kwargs) -> Array:
        start = time.perf_counter()
        out = fun(*args, **kwargs)

        def _block(x):
            if isinstance(x, Array):
                x.block_until_ready()

        jax.tree.map(_block, out)
        end = time.perf_counter()
        self.jit_time = (end - start) * 1e3

        self.compiled_code['JAXPR'] = 'N/A'
        self.compiled_code['LOWERED'] = 'N/A'
        self.compiled_code['COMPILED'] = 'N/A'
        self.profiling_data['generated_code'] = 'N/A'
        self.profiling_data['argument_size'] = 'N/A'
        self.profiling_data['output_size'] = 'N/A'
        self.profiling_data['temp_size'] = 'N/A'

        if self.save_jaxpr:
            jaxpr = make_jaxpr(fun, static_argnums=self.static_argnums)(*args, **kwargs)
            self.compiled_code['JAXPR'] = jaxpr.pretty_print()

        if self.compile_info:
            lowered = jax.jit(fun, static_argnums=self.static_argnums).lower(*args, **kwargs)
            compiled = lowered.compile()
            memory_analysis = self._read_memory_analysis(compiled.memory_analysis())

            self.compiled_code['LOWERED'] = lowered.as_text()
            self.compiled_code['COMPILED'] = compiled.as_text()
            self.profiling_data['generated_code'] = memory_analysis[0]
            self.profiling_data['argument_size'] = memory_analysis[1]
            self.profiling_data['output_size'] = memory_analysis[2]
            self.profiling_data['temp_size'] = memory_analysis[3]

        return out

    def chrono_fun(self, fun: Callable, *args, **kwargs) -> Array:
        start = time.perf_counter()
        out = fun(*args, **kwargs)

        def _block(x):
            if isinstance(x, Array):
                x.block_until_ready()

        jax.tree.map(_block, out)
        end = time.perf_counter()
        self.times.append((end - start) * 1e3)
        return out

    def _get_mean_times(self) -> np.ndarray:
        if jax.device_count() == 1 or jax.process_count() == 1:
            return np.array(self.times)

        if self.devices is None:
            self.devices = jax.devices()

        mesh = jax.make_mesh((len(self.devices),), ('x',), devices=self.devices)
        sharding = NamedSharding(mesh, P('x'))

        times_array = jnp.array(self.times)
        global_shape = (jax.device_count(), times_array.shape[0])
        global_times = jax.make_array_from_callback(
            shape=global_shape,
            sharding=sharding,
            data_callback=lambda _: jnp.expand_dims(times_array, axis=0),
        )

        @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P())
        def get_mean_times(times) -> Array:
            return jax.lax.pmean(times, axis_name='x')

        times_array = get_mean_times(global_times)
        times_array.block_until_ready()
        return np.array(times_array.addressable_data(0)[0])
