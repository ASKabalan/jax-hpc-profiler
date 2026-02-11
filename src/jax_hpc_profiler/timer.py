import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
from tabulate import tabulate


class AbstractTimer(ABC):
    def __init__(self):
        self.jit_time = 0.0
        self.times = []
        self.profiling_data = {
            'generated_code': 'N/A',
            'argument_size': 'N/A',
            'output_size': 'N/A',
            'temp_size': 'N/A',
        }
        self.compiled_code = {'JAXPR': 'N/A', 'LOWERED': 'N/A', 'COMPILED': 'N/A'}

    @abstractmethod
    def chrono_jit(self, fun: Callable, *args, **kwargs): ...

    @abstractmethod
    def chrono_fun(self, fun: Callable, *args, **kwargs): ...

    @abstractmethod
    def _get_mean_times(self) -> np.ndarray: ...

    def _should_write(self) -> bool:
        return True

    def _has_compile_info(self) -> bool:
        return False

    @staticmethod
    def _normalize_memory_units(memory_analysis) -> str:
        if memory_analysis == 'N/A':
            return memory_analysis

        sizes_str = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        factors = [1, 1024, 1024**2, 1024**3, 1024**4, 1024**5]
        factor = 0 if memory_analysis == 0 else int(np.log10(memory_analysis) // 3)

        return f'{memory_analysis / factors[factor]:.2f} {sizes_str[factor]}'

    def _reset(self):
        self.jit_time = 0.0
        self.times = []
        self.profiling_data = {
            'generated_code': 'N/A',
            'argument_size': 'N/A',
            'output_size': 'N/A',
            'temp_size': 'N/A',
        }
        self.compiled_code = {'JAXPR': 'N/A', 'LOWERED': 'N/A', 'COMPILED': 'N/A'}

    def report(
        self,
        csv_filename: str,
        function: str,
        x: int,
        y: int | None = None,
        z: int | None = None,
        precision: str = 'float32',
        px: int = 1,
        py: int = 1,
        backend: str = 'NCCL',
        nodes: int = 1,
        md_filename: str | None = None,
        npz_data: Optional[dict[str, Any]] = None,
        extra_info: dict[str, Any] = {},
    ) -> None:
        if self.jit_time == 0.0 and len(self.times) == 0:
            print(f'No profiling data to report for {function}')
            self._reset()
            return

        if md_filename is None:
            dirname, filename = (
                os.path.dirname(csv_filename),
                os.path.splitext(os.path.basename(csv_filename))[0],
            )
            report_folder = filename if dirname == '' else f'{dirname}/{filename}'
            os.makedirs(report_folder, exist_ok=True)
            md_filename = f'{report_folder}/{x}_{px}_{py}_{backend}_{precision}_{function}.md'

        if npz_data is not None:
            dirname, filename = (
                os.path.dirname(csv_filename),
                os.path.splitext(os.path.basename(csv_filename))[0],
            )
            report_folder = filename if dirname == '' else f'{dirname}/{filename}'
            os.makedirs(report_folder, exist_ok=True)
            npz_filename = f'{report_folder}/{x}_{px}_{py}_{backend}_{precision}_{function}.npz'
            np.savez(npz_filename, **npz_data)

        y = x if y is None else y
        z = x if z is None else z

        times_array = self._get_mean_times()
        if self._should_write():
            min_time = np.min(times_array)
            max_time = np.max(times_array)
            mean_time = np.mean(times_array)
            std_time = np.std(times_array)
            last_time = times_array[-1]
            generated_code = self.profiling_data.get('generated_code', 'N/A')
            argument_size = self.profiling_data.get('argument_size', 'N/A')
            output_size = self.profiling_data.get('output_size', 'N/A')
            temp_size = self.profiling_data.get('temp_size', 'N/A')

            csv_line = (
                f'{function},{precision},{x},{y},{z},{px},{py},{backend},{nodes},'
                f'{self.jit_time:.4f},{min_time:.4f},{max_time:.4f},{mean_time:.4f},{std_time:.4f},{last_time:.4f},'
                f'{generated_code},{argument_size},{output_size},{temp_size}\n'
            )

            with open(csv_filename, 'a') as f:
                f.write(csv_line)

            param_dict = {
                'Function': function,
                'Precision': precision,
                'X': x,
                'Y': y,
                'Z': z,
                'PX': px,
                'PY': py,
                'Backend': backend,
                'Nodes': nodes,
            }
            param_dict.update(extra_info)
            profiling_result = {
                'JIT Time': self.jit_time,
                'Min Time': min_time,
                'Max Time': max_time,
                'Mean Time': mean_time,
                'Std Time': std_time,
                'Last Time': last_time,
                'Generated Code': self._normalize_memory_units(generated_code),
                'Argument Size': self._normalize_memory_units(argument_size),
                'Output Size': self._normalize_memory_units(output_size),
                'Temporary Size': self._normalize_memory_units(temp_size),
            }
            iteration_runs = {}
            for i in range(len(times_array)):
                iteration_runs[f'Run {i}'] = times_array[i]

            with open(md_filename, 'w') as f:
                f.write(f'# Reporting for {function}\n')
                f.write('## Parameters\n')
                f.write(
                    tabulate(
                        param_dict.items(),
                        headers=['Parameter', 'Value'],
                        tablefmt='github',
                    )
                )
                f.write('\n---\n')
                f.write('## Profiling Data\n')
                f.write(
                    tabulate(
                        profiling_result.items(),
                        headers=['Parameter', 'Value'],
                        tablefmt='github',
                    )
                )
                f.write('\n---\n')
                f.write('## Iteration Runs\n')
                f.write(
                    tabulate(
                        iteration_runs.items(),
                        headers=['Iteration', 'Time'],
                        tablefmt='github',
                    )
                )
                if self._has_compile_info():
                    f.write('\n---\n')
                    f.write('## Compiled Code\n')
                    f.write('```hlo\n')
                    f.write(self.compiled_code['COMPILED'])
                    f.write('\n```\n')
                    f.write('\n---\n')
                    f.write('## Lowered Code\n')
                    f.write('```hlo\n')
                    f.write(self.compiled_code['LOWERED'])
                    f.write('\n```\n')
                    f.write('\n---\n')
                    if self.compiled_code.get('JAXPR', 'N/A') != 'N/A':
                        f.write('## JAXPR\n')
                        f.write('```haskel\n')
                        f.write(self.compiled_code['JAXPR'])
                        f.write('\n```\n')

        self._reset()


class NoTimer(AbstractTimer):
    def chrono_jit(self, fun: Callable, *args, **kwargs):
        return fun(*args, **kwargs)

    def chrono_fun(self, fun: Callable, *args, **kwargs):
        return fun(*args, **kwargs)

    def _get_mean_times(self) -> np.ndarray:
        return np.array([])

    def _should_write(self) -> bool:
        return False

    def report(self, *args, **kwargs) -> None:
        pass


def Timer(
    save_jaxpr=False,
    compile_info=True,
    jax_fn=True,
    active=True,
    devices=None,
    static_argnums=(),
):
    if not active:
        return NoTimer()
    if jax_fn:
        from .jax_timer import JaxTimer

        return JaxTimer(
            save_jaxpr=save_jaxpr,
            compile_info=compile_info,
            devices=devices,
            static_argnums=static_argnums,
        )
    from .numpy_timer import NumpyTimer

    return NumpyTimer()
