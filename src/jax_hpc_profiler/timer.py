import os
import time
from functools import partial
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tabulate import tabulate


class Timer:

    def __init__(self, save_jaxpr=False):
        self.jit_time = None
        self.times = []
        self.profiling_data = {}
        self.compiled_code = {}
        self.save_jaxpr = save_jaxpr

    def chrono_jit(self, fun: Callable, *args, ndarray_arg=None) -> np.ndarray:
        start = time.perf_counter()
        out = jax.jit(fun)(*args)
        if ndarray_arg is None:
            out.block_until_ready()
        else:
            out[ndarray_arg].block_until_ready()
        end = time.perf_counter()
        self.jit_time = (end - start) * 1e3

        if self.save_jaxpr:
            jaxpr = make_jaxpr(fun)(*args)
            self.compiled_code["JAXPR"] = jaxpr.pretty_print()

        lowered = jax.jit(fun).lower(*args)
        compiled = lowered.compile()
        memory_analysis = compiled.memory_analysis()
        self.compiled_code["LOWERED"] = lowered.as_text()
        self.compiled_code["COMPILED"] = compiled.as_text()
        self.profiling_data["FLOPS"] = compiled.cost_analysis()[0]['flops']
        self.profiling_data[
            "generated_code"] = memory_analysis.generated_code_size_in_bytes
        self.profiling_data[
            "argument_size"] = memory_analysis.argument_size_in_bytes
        self.profiling_data[
            "output_size"] = memory_analysis.output_size_in_bytes
        self.profiling_data["temp_size"] = memory_analysis.temp_size_in_bytes

        return out

    def chrono_fun(self, fun: Callable, *args, ndarray_arg=None) -> np.ndarray:
        start = time.perf_counter()
        out = fun(*args)
        if ndarray_arg is None:
            out.block_until_ready()
        else:
            out[ndarray_arg].block_until_ready()
        end = time.perf_counter()
        self.times.append((end - start) * 1e3)
        return out

    def _get_mean_times(self, times_array: jnp.ndarray,
                        sharding: NamedSharding):
        mesh = sharding.mesh
        specs = sharding.spec
        valid_letters = [letter for letter in specs if letter is not None]
        assert len(valid_letters
                   ) > 0, "Sharding was provided but with no partition specs"

        @partial(shard_map,
                 mesh=mesh,
                 in_specs=specs,
                 out_specs=P(),
                 check_rep=False)
        def get_mean_times(times):
            mean = jax.lax.pmean(times, axis_name=valid_letters[0])
            for axis_name in valid_letters[1:]:
                mean = jax.lax.pmean(mean, axis_name=axis_name)
            return mean

        times_array = get_mean_times(times_array)
        times_array.block_until_ready()
        return times_array

    def report(self,
               csv_filename: str,
               function: str,
               precision: str,
               x: int,
               y: int,
               z: int,
               px: int,
               py: int,
               backend: str,
               nodes: int,
               sharding: NamedSharding | None = None,
               md_filename: str | None = None,
               extra_info: dict = {}):
        times_array = jnp.array(self.times)

        if md_filename is None:
            dirname, filename = os.path.dirname(csv_filename), os.path.splitext(os.path.basename(csv_filename))[0]
            report_folder = filename if dirname == "" else f"{dirname}/{filename}"
            print(f"report_folder: {report_folder} csv_filename: {csv_filename}")
            os.makedirs(report_folder, exist_ok=True)
            md_filename = f"{report_folder}/{x}_{px}_{py}_{backend}_{precision}_{function}.md"

        if sharding is not None:
            times_array = self._get_mean_times(times_array, sharding)

        times_array = np.array(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        last_time = times_array[-1]

        flops = self.profiling_data["FLOPS"]
        generated_code = self.profiling_data["generated_code"]
        argument_size = self.profiling_data["argument_size"]
        output_size = self.profiling_data["output_size"]
        temp_size = self.profiling_data["temp_size"]

        csv_line = (
            f"{function},{precision},{x},{y},{z},{px},{py},{backend},{nodes},"
            f"{self.jit_time:.4f},{min_time:.4f},{max_time:.4f},{mean_time:.4f},{std_time:.4f},{last_time:.4f},"
            f"{generated_code},{argument_size},{output_size},{temp_size},{flops}\n"
        )

        with open(csv_filename, 'a') as f:
            f.write(csv_line)

        param_dict = {
            "Function": function,
            "Precision": precision,
            "X": x,
            "Y": y,
            "Z": z,
            "PX": px,
            "PY": py,
            "Backend": backend,
            "Nodes": nodes,
        }
        param_dict.update(extra_info)
        profiling_result = {
            "JIT Time": self.jit_time,
            "Min Time": min_time,
            "Max Time": max_time,
            "Mean Time": mean_time,
            "Std Time": std_time,
            "Last Time": last_time,
            "Generated Code": generated_code,
            "Argument Size": argument_size,
            "Output Size": output_size,
            "Temporary Size": temp_size,
            "FLOPS": self.profiling_data["FLOPS"]
        }

        with open(md_filename, 'w') as f:
            f.write(f"# Reporting for {function}\n")
            f.write(f"## Parameters\n")
            f.write(tabulate(param_dict.items() , headers=["Parameter" , "Value"] , tablefmt='github'))
            f.write("\n---\n")
            f.write(f"## Profiling Data\n")
            f.write(tabulate(profiling_result.items() , headers=["Parameter" , "Value"] , tablefmt='github'))
            f.write("\n---\n")
            f.write(f"## Compiled Code\n")
            f.write(f"```hlo\n")
            f.write(self.compiled_code["COMPILED"])
            f.write(f"\n```\n")
            f.write("\n---\n")
            f.write(f"## Lowered Code\n")
            f.write(f"```hlo\n")
            f.write(self.compiled_code["LOWERED"])
            f.write(f"\n```\n")
            f.write("\n---\n")
            if self.save_jaxpr:
                f.write(f"## JAXPR\n")
                f.write(f"```haskel\n")
                f.write(self.compiled_code["JAXPR"])
                f.write(f"\n```\n")
