from itertools import product
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from .utils import clean_up_csv, plot_with_pdims_strategy

np.seterr(divide='ignore')


def configure_axes(
    ax: Axes,
    x_values: List[int],
    y_values: List[float],
    title: Optional[str],
    xlabel: str,
    plotting_memory: bool = False,
    memory_units: str = 'bytes',
):
    """
    Configure the axes for the plot.

    Parameters
    ----------
    ax : Axes
        The axes to configure.
    x_values : List[int]
        The x-axis values.
    y_values : List[float]
        The y-axis values.
    xlabel : str
        The label for the x-axis.
    """
    ylabel = 'Time (milliseconds)' if not plotting_memory else f'Memory ({memory_units})'

    def f2(x):
        return np.log2(x)

    def g2(x):
        return 2**x

    ax.set_xlim([min(x_values), max(x_values)])
    y_min, y_max = min(y_values) * 0.6, max(y_values) * 1.1
    ax.set_title(title)
    ax.set_ylim([y_min, y_max])
    ax.set_xscale('function', functions=(f2, g2))
    if not plotting_memory:
        ax.set_yscale('symlog')
        time_ticks = [
            10**t for t in range(int(np.floor(np.log10(y_min))), 1 + int(np.ceil(np.log10(y_max))))
        ]
        ax.set_yticks(time_ticks)
    ax.set_xticks(x_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for x_value in x_values:
        ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)
    ax.legend(
        loc='best',
    )


def plot_scaling(
    dataframes: Dict[str, pd.DataFrame],
    fixed_sizes: List[int],
    size_column: str,
    fixed_column: str,
    xlabel: str,
    title: str,
    figure_size: tuple = (6, 4),
    output: Optional[str] = None,
    dark_bg: bool = False,
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = 'plot',
    pdims_strategy: List[str] = ['plot_fastest'],
):
    """
    General scaling plot function based on the number of GPUs or data size.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    fixed_sizes : List[int]
        List of fixed sizes (data or GPUs) to plot.
    size_column : str
        Column name for the size axis ('x' for weak scaling, 'gpus' for strong scaling).
    fixed_column : str
        Column name for the fixed axis ('gpus' for weak scaling, 'x' for strong scaling).
    xlabel : str
        Label for the x-axis.
    figure_size : tuple, optional
        Size of the figure, by default (6, 4).
    output : Optional[str], optional
        Output file to save the plot, by default None.
    dark_bg : bool, optional
        Whether to use dark background for the plot, by default False.
    print_decompositions : bool, optional
        Whether to print decompositions on the plot, by default False.
    backends : Optional[List[str]], optional
        List of backends to include, by default None.
    pdims_strategy : str, optional
        Strategy for plotting pdims ('plot_all' or 'plot_fastest'), by default 'plot_fastest'.
    """

    if dark_bg:
        plt.style.use('dark_background')

    num_subplots = len(fixed_sizes)
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, fixed_size in enumerate(fixed_sizes):
        ax: Axes = axs[i]

        x_values = []
        y_values = []
        for method, df in dataframes.items():
            filtered_method_df = df[df[fixed_column] == int(fixed_size)]
            if filtered_method_df.empty:
                continue
            filtered_method_df = filtered_method_df.sort_values(by=[size_column])
            functions = (
                pd.unique(filtered_method_df['function']) if functions is None else functions
            )
            precisions = (
                pd.unique(filtered_method_df['precision']) if precisions is None else precisions
            )
            backends = pd.unique(filtered_method_df['backend']) if backends is None else backends

            combinations = product(backends, precisions, functions, plot_columns)

            for backend, precision, function, plot_column in combinations:
                filtered_params_df = filtered_method_df[
                    (filtered_method_df['backend'] == backend)
                    & (filtered_method_df['precision'] == precision)
                    & (filtered_method_df['function'] == function)
                ]
                if filtered_params_df.empty:
                    continue
                x_vals, y_vals = plot_with_pdims_strategy(
                    ax,
                    filtered_params_df,
                    method,
                    pdims_strategy,
                    print_decompositions,
                    size_column,
                    plot_column,
                    label_text,
                )

                x_values.extend(x_vals)
                y_values.extend(y_vals)

        if len(x_values) != 0:
            plotting_memory = 'time' not in plot_columns[0].lower()
            figure_title = f'{title} {fixed_size}' if title is not None else None
            configure_axes(
                ax,
                x_values,
                y_values,
                figure_title,
                xlabel,
                plotting_memory,
                memory_units,
            )

    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    fig.tight_layout()
    rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='round,pad=0.02', ec='black', fc='none')
    fig.patches.append(rect)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def plot_strong_scaling(
    csv_files: List[str],
    fixed_gpu_size: Optional[List[int]] = None,
    fixed_data_size: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ['plot_fastest'],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = '%m%-%f%-%pn%-%pr%-%b%-%p%-%n%',
    xlabel: str = 'Number of GPUs',
    title: str = 'Data sizes',
    figure_size: tuple = (6, 4),
    dark_bg: bool = False,
    output: Optional[str] = None,
):
    """
    Plot strong scaling based on the number of GPUs.
    """

    dataframes, _, available_data_sizes = clean_up_csv(
        csv_files,
        precisions,
        functions,
        fixed_gpu_size,
        fixed_data_size,
        pdims,
        pdims_strategy,
        backends,
        memory_units,
    )
    if len(dataframes) == 0:
        print('No dataframes found for the given arguments. Exiting...')
        return

    plot_scaling(
        dataframes,
        available_data_sizes,
        'gpus',
        'x',
        xlabel,
        title,
        figure_size,
        output,
        dark_bg,
        print_decompositions,
        backends,
        precisions,
        functions,
        plot_columns,
        memory_units,
        label_text,
        pdims_strategy,
    )


def plot_weak_scaling(
    csv_files: List[str],
    fixed_gpu_size: Optional[List[int]] = None,
    fixed_data_size: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ['plot_fastest'],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = '%m%-%f%-%pn%-%pr%-%b%-%p%-%n%',
    xlabel: str = 'Number of GPUs',
    title: Optional[str] = None,
    figure_size: tuple = (6, 4),
    dark_bg: bool = False,
    output: Optional[str] = None,
    ideal_line: bool = False,
    reverse_axes: bool = False,
):
    """
    Plot true weak scaling: runtime vs GPUs for explicit (gpus, data size) sequences.

    Both ``fixed_gpu_size`` and ``fixed_data_size`` must be provided and have the same length,
    representing explicit weak-scaling pairs (gpus[i], data_size[i]).

    reverse_axes:
        - False (default): x-axis is GPUs, y-axis is time; points are annotated with ``N=<data_size>``.
        - True: x-axis is data size, y-axis is time; points are annotated with ``GPUs=<gpu_count>``.
    """
    if fixed_gpu_size is None or fixed_data_size is None:
        raise ValueError(
            'Weak scaling requires both fixed_gpu_size (gpus) and fixed_data_size (problem sizes).'
        )
    if len(fixed_gpu_size) != len(fixed_data_size):
        raise ValueError(
            'Weak scaling requires fixed_gpu_size and fixed_data_size lists of equal length.'
        )

    gpu_to_data = {int(g): int(d) for g, d in zip(fixed_gpu_size, fixed_data_size)}
    data_to_gpu = {int(d): int(g) for g, d in zip(fixed_gpu_size, fixed_data_size)}
    x_col = 'x' if reverse_axes else 'gpus'

    dataframes, _, _ = clean_up_csv(
        csv_files,
        precisions,
        functions,
        fixed_gpu_size,
        fixed_data_size,
        pdims,
        pdims_strategy,
        backends,
        memory_units,
    )
    if len(dataframes) == 0:
        print('No dataframes found for the given arguments. Exiting...')
        return

    if dark_bg:
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=figure_size)

    x_values: List[float] = []
    y_values: List[float] = []
    annotations: List = []
    ideal_line_plotted = False

    for method, df in dataframes.items():
        # Determine parameter sets from the filtered dataframe if not provided
        local_functions = pd.unique(df['function']) if functions is None else functions
        local_precisions = pd.unique(df['precision']) if precisions is None else precisions
        local_backends = pd.unique(df['backend']) if backends is None else backends

        combinations = product(local_backends, local_precisions, local_functions, plot_columns)

        for backend, precision, function, plot_column in combinations:
            base_df = df[
                (df['backend'] == backend)
                & (df['precision'] == precision)
                & (df['function'] == function)
            ]
            if base_df.empty:
                continue

            # Keep only rows matching any of the (gpus, x) pairs
            mask = pd.Series(False, index=base_df.index)
            for g, d in zip(fixed_gpu_size, fixed_data_size):
                mask |= (base_df['gpus'] == int(g)) & (base_df['x'] == int(d))

            filtered_params_df = base_df[mask]
            if filtered_params_df.empty:
                continue

            x_vals, y_vals = plot_with_pdims_strategy(
                ax,
                filtered_params_df,
                method,
                pdims_strategy,
                print_decompositions,
                x_col,
                plot_column,
                label_text,
            )
            if x_vals is None or len(x_vals) == 0:
                continue

            x_arr = np.asarray(x_vals).reshape(-1)
            y_arr = np.asarray(y_vals).reshape(-1)

            # Annotate every point with data size or GPU count depending on axis choice.
            # Use plain data coordinates for the text; adjust_text will then only move
            # the labels slightly (mostly vertically) to avoid overlap.
            for xv, yv in zip(x_arr, y_arr):
                if reverse_axes:
                    gpu = data_to_gpu.get(int(xv))
                    if gpu is None:
                        continue
                    label = f'GPUs={gpu}'
                else:
                    data_size = gpu_to_data.get(int(xv))
                    if data_size is None:
                        continue
                    label = f'N={data_size}'

                text_obj = ax.text(
                    float(xv),
                    float(yv),
                    label,
                    ha='center',
                    va='bottom',
                    fontsize='small',
                    clip_on=True,
                )
                annotations.append(text_obj)

            x_values.extend(x_arr.tolist())
            y_values.extend(y_arr.tolist())

            if ideal_line and not ideal_line_plotted:
                # Use the smallest x value in this curve as baseline
                baseline_index = np.argmin(x_arr)
                baseline_y = y_arr[baseline_index]
                ax.hlines(
                    baseline_y,
                    xmin=float(np.min(x_arr)),
                    xmax=float(np.max(x_arr)),
                    colors='gray',
                    linestyles='dashed',
                    label='Ideal weak scaling',
                )
                ideal_line_plotted = True
                y_values.append(float(baseline_y))

    if x_values:
        plotting_memory = 'time' not in plot_columns[0].lower()
        figure_title = title if title is not None else 'Weak scaling'
        configure_axes(
            ax,
            x_values,
            y_values,
            figure_title,
            xlabel,
            plotting_memory,
            memory_units,
        )
        if annotations:
            ax.figure.canvas.draw()
            adjust_text(
                annotations,
                ax=ax,
                # keep points aligned in x, only allow vertical motion
                only_move={'text': 'y', 'static': 'y'},
                expand=(1.02, 1.05),
                force_text=(0.08, 0.2),
                max_move=(0, 30),
            )

    fig.tight_layout()
    rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle='round,pad=0.02', ec='black', fc='none')
    fig.patches.append(rect)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def plot_weak_fixed_scaling(
    csv_files: List[str],
    fixed_gpu_size: Optional[List[int]] = None,
    fixed_data_size: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ['plot_fastest'],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = '%m%-%f%-%pn%-%pr%-%b%-%p%-%n%',
    xlabel: str = 'Data sizes',
    title: str = 'Number of GPUs',
    figure_size: tuple = (6, 4),
    dark_bg: bool = False,
    output: Optional[str] = None,
):
    """
    Plot size scaling at fixed GPU count (previous weak-scaling behavior).
    """
    dataframes, available_gpu_counts, _ = clean_up_csv(
        csv_files,
        precisions,
        functions,
        fixed_gpu_size,
        fixed_data_size,
        pdims,
        pdims_strategy,
        backends,
        memory_units,
    )
    if len(dataframes) == 0:
        print('No dataframes found for the given arguments. Exiting...')
        return

    plot_scaling(
        dataframes,
        available_gpu_counts,
        'x',
        'gpus',
        xlabel,
        title,
        figure_size,
        output,
        dark_bg,
        print_decompositions,
        backends,
        precisions,
        functions,
        plot_columns,
        memory_units,
        label_text,
        pdims_strategy,
    )
