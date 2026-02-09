from itertools import product
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from .utils import _query_volume_type, clean_up_csv, plot_with_pdims_strategy

np.seterr(divide='ignore')


def _format_volume_title(vol: int) -> str:
    """Format a volume as N³ if it is a perfect cube, otherwise as the raw number."""
    cbrt = round(vol ** (1.0 / 3.0))
    if cbrt**3 == vol:
        return f'{cbrt}\u00b3'
    return f'{vol:,}'


def _build_volume_labels(
    dataframes: Dict[str, pd.DataFrame],
    vol_column: str,
    volumes: List[int],
) -> tuple[dict[int, str], bool]:
    """Build tick labels for volume x-axis values.

    Returns (labels_dict, has_non_cube) where labels_dict maps volume → label string,
    and has_non_cube is True if any label is not a simple N³ (needs tick rotation).
    """
    labels = {}
    has_non_cube = False
    for vol in sorted(set(int(v) for v in volumes)):
        cbrt = round(vol ** (1.0 / 3.0))
        if cbrt**3 == vol:
            labels[vol] = f'{cbrt}\u00b3'
        elif vol_column == 'global_vol':
            # Look up first (x, y, z) row for this volume
            found = False
            for df in dataframes.values():
                rows = df[df[vol_column] == vol]
                if not rows.empty:
                    row = rows.iloc[0]
                    x, y, z = int(row['x']), int(row['y']), int(row['z'])
                    if x == y == z:
                        labels[vol] = f'{x}\u00b3'
                    else:
                        labels[vol] = f'{x}\u00d7{y}\u00d7{z}'
                        has_non_cube = True
                    found = True
                    break
            if not found:
                labels[vol] = f'{vol:,}'
                has_non_cube = True
        else:
            labels[vol] = f'{vol:,}'
            has_non_cube = True
    return labels, has_non_cube


def configure_axes(
    ax: Axes,
    x_values: List[int],
    y_values: List[float],
    title: Optional[str],
    xlabel: str,
    plotting_memory: bool = False,
    memory_units: str = 'bytes',
    xscale: str = 'linear',
    x_tick_labels: Optional[dict] = None,
    rotate_x_ticks: bool = False,
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
    title : Optional[str]
        The title for the subplot.
    xlabel : str
        The label for the x-axis.
    plotting_memory : bool
        Whether we are plotting memory instead of time.
    memory_units : str
        The memory unit label.
    xscale : str
        X-axis scale: 'linear', 'symlog', 'log2', or 'log10'.
    x_tick_labels : Optional[dict]
        Mapping from x-value to custom tick label string.
    rotate_x_ticks : bool
        Whether to rotate x-axis tick labels by 45 degrees.
    """
    ylabel = 'Time (milliseconds)' if not plotting_memory else f'Memory ({memory_units})'

    ax.set_xlim([min(x_values), max(x_values)])
    y_min, y_max = min(y_values) * 0.6, max(y_values) * 1.1
    ax.set_title(title)
    ax.set_ylim([y_min, y_max])
    if not plotting_memory:
        ax.set_yscale('symlog')
        time_ticks = [
            10**t for t in range(int(np.floor(np.log10(y_min))), 1 + int(np.ceil(np.log10(y_max))))
        ]
        ax.set_yticks(time_ticks)

    # Apply x-axis scale
    if xscale == 'log2':
        ax.set_xscale('function', functions=(np.log2, lambda x: 2**x))
    elif xscale == 'log10':
        ax.set_xscale('function', functions=(np.log10, lambda x: 10**x))
    elif xscale == 'symlog':
        ax.set_xscale('symlog')
    # else: linear (default, no call needed)

    ax.set_xticks(x_values)
    if x_tick_labels:
        ax.set_xticklabels([x_tick_labels.get(int(v), str(v)) for v in x_values])
    else:
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    if rotate_x_ticks:
        ax.tick_params(axis='x', rotation=45)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for x_value in x_values:
        ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)
    ax.legend(
        loc='best',
    )


def plot_scaling(
    dataframes: Dict[str, pd.DataFrame],
    scaling_labels: List,
    scaling_label_column: str,
    x_column: str,
    xlabel: str,
    title: str,
    figure_size: tuple = (6, 4),
    output: Optional[str] = None,
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = 'plot',
    pdims_strategy: List[str] = ['plot_fastest'],
    ideal_line: bool = False,
    xscale: str = 'linear',
):
    """
    General scaling plot function.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    scaling_labels : List
        List of values to create subplots for.
    scaling_label_column : str
        Column name for the subplot axis.
    x_column : str
        Column name for the x-axis ('gpus', 'global_vol', or 'local_vol').
    xlabel : str
        Label for the x-axis.
    title : str
        Base title for the plot.
    figure_size : tuple, optional
        Size of the figure, by default (6, 4).
    output : Optional[str], optional
        Output file to save the plot, by default None.
    print_decompositions : bool, optional
        Whether to print decompositions on the plot, by default False.
    backends : Optional[List[str]], optional
        List of backends to include, by default None.
    precisions : Optional[List[str]], optional
        List of precisions to include, by default None.
    functions : Optional[List[str]], optional
        List of functions to include, by default None.
    plot_columns : List[str], optional
        Columns to plot, by default ['mean_time'].
    memory_units : str, optional
        Memory unit label, by default 'bytes'.
    label_text : str, optional
        Template for plot labels, by default 'plot'.
    pdims_strategy : List[str], optional
        Strategy for plotting pdims, by default ['plot_fastest'].
    ideal_line : bool, optional
        Whether to draw an ideal scaling reference line, by default False.
        Only drawn when x_column is 'gpus': global_vol → 1/N, local_vol → flat.
    xscale : str, optional
        X-axis scale: 'linear', 'symlog', 'log2', or 'log10', by default 'linear'.
    """
    num_subplots = len(scaling_labels)
    if num_subplots == 0:
        print('No volumes to plot. Exiting...')
        return
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, label_value in enumerate(scaling_labels):
        ax: Axes = axs[i]

        x_values = []
        y_values = []
        ideal_line_plotted = False

        for method, df in dataframes.items():
            filtered_method_df = df[df[scaling_label_column] == int(label_value)]
            if filtered_method_df.empty:
                continue
            filtered_method_df = filtered_method_df.sort_values(by=[x_column])
            local_functions = (
                pd.unique(filtered_method_df['function']) if functions is None else functions
            )
            local_precisions = (
                pd.unique(filtered_method_df['precision']) if precisions is None else precisions
            )
            local_backends = (
                pd.unique(filtered_method_df['backend']) if backends is None else backends
            )

            combinations = product(local_backends, local_precisions, local_functions, plot_columns)

            for backend, precision, function, plot_column in combinations:
                filtered_params_df = filtered_method_df[
                    (filtered_method_df['backend'] == backend)
                    & (filtered_method_df['precision'] == precision)
                    & (filtered_method_df['function'] == function)
                ]
                if filtered_params_df.empty:
                    continue
                result = plot_with_pdims_strategy(
                    ax,
                    filtered_params_df,
                    method,
                    pdims_strategy,
                    print_decompositions,
                    x_column,
                    plot_column,
                    label_text,
                )
                if result is None:
                    continue

                x_vals, y_vals = result
                x_values.extend(x_vals)
                y_values.extend(y_vals)

                # Draw ideal line once per subplot, anchored at first curve
                # Only meaningful when x-axis is GPUs
                if (
                    ideal_line
                    and not ideal_line_plotted
                    and len(x_vals) > 0
                    and x_column == 'gpus'
                ):
                    x_arr = np.asarray(x_vals).reshape(-1)
                    y_arr = np.asarray(y_vals).reshape(-1)
                    baseline_idx = np.argmin(x_arr)
                    baseline_gpus = x_arr[baseline_idx]
                    baseline_y = y_arr[baseline_idx]
                    sorted_x = sorted(set(x_values))

                    if scaling_label_column == 'global_vol':
                        # Strong scaling ideal: T(N) = T(N_min) * N_min / N
                        ideal_y = [baseline_y * baseline_gpus / g for g in sorted_x]
                        ax.plot(sorted_x, ideal_y, '--', color='gray', label='Ideal scaling')
                    else:
                        # Weak scaling ideal: T(N) = T(N_min) (flat)
                        ax.hlines(
                            baseline_y,
                            min(sorted_x),
                            max(sorted_x),
                            colors='gray',
                            linestyles='dashed',
                            label='Ideal scaling',
                        )
                    ideal_line_plotted = True

        if len(x_values) != 0:
            plotting_memory = 'time' not in plot_columns[0].lower()
            vol_label = _format_volume_title(int(label_value))
            figure_title = f'{title} {vol_label}' if title is not None else None

            # Build volume tick labels when x-axis is a volume column
            tick_labels = None
            rotate = False
            if x_column in ('global_vol', 'local_vol'):
                tick_labels, has_non_cube = _build_volume_labels(
                    dataframes, x_column, x_values
                )
                rotate = has_non_cube

            configure_axes(
                ax,
                x_values,
                y_values,
                figure_title,
                xlabel,
                plotting_memory,
                memory_units,
                xscale,
                tick_labels,
                rotate,
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


def plot_by_data_size(
    csv_files: List[str],
    gpus: Optional[List[int]] = None,
    data_size_queries: Optional[List[str]] = None,
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
    output: Optional[str] = None,
    ideal_line: bool = False,
    xscale: str = 'linear',
):
    """
    Plot with subplots per data size query, x-axis = GPUs.

    Auto-detects volume column from queries:
    - ``global_*`` queries → subplots by ``global_vol``, ideal = 1/N (strong scaling)
    - ``local_*`` queries → subplots by ``local_vol``, ideal = flat (weak scaling)
    """
    dataframes, _, _ = clean_up_csv(
        csv_files,
        precisions,
        functions,
        gpus,
        data_size_queries,
        pdims,
        pdims_strategy,
        backends,
        memory_units,
    )
    if len(dataframes) == 0:
        print('No dataframes found for the given arguments. Exiting...')
        return

    # Auto-detect volume column from queries
    if data_size_queries:
        vol_col = _query_volume_type(data_size_queries[0])
    else:
        vol_col = 'global_vol'

    volumes = sorted({v for df in dataframes.values() for v in df[vol_col].unique()})

    plot_scaling(
        dataframes,
        volumes,
        vol_col,
        'gpus',
        xlabel,
        title,
        figure_size,
        output,
        print_decompositions,
        backends,
        precisions,
        functions,
        plot_columns,
        memory_units,
        label_text,
        pdims_strategy,
        ideal_line,
        xscale,
    )


def plot_by_gpus(
    csv_files: List[str],
    gpus: Optional[List[int]] = None,
    data_size_queries: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ['plot_fastest'],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ['mean_time'],
    memory_units: str = 'bytes',
    label_text: str = '%m%-%f%-%pn%-%pr%-%b%-%p%-%n%',
    xlabel: str = 'Data size',
    title: str = 'GPU counts',
    figure_size: tuple = (6, 4),
    output: Optional[str] = None,
    ideal_line: bool = False,
    xscale: str = 'linear',
):
    """
    Plot with subplots per GPU count, x-axis = data size (volume).

    Auto-detects volume column from queries:
    - ``local_*`` queries → x-axis = ``local_vol``
    - ``global_*`` queries (default) → x-axis = ``global_vol``
    """
    dataframes, available_gpus, _ = clean_up_csv(
        csv_files,
        precisions,
        functions,
        gpus,
        data_size_queries,
        pdims,
        pdims_strategy,
        backends,
        memory_units,
    )
    if len(dataframes) == 0:
        print('No dataframes found for the given arguments. Exiting...')
        return

    # Auto-detect volume column from queries
    if data_size_queries:
        vol_col = _query_volume_type(data_size_queries[0])
    else:
        vol_col = 'global_vol'

    gpu_list = sorted(available_gpus)

    plot_scaling(
        dataframes,
        gpu_list,
        'gpus',
        vol_col,
        xlabel,
        title,
        figure_size,
        output,
        print_decompositions,
        backends,
        precisions,
        functions,
        plot_columns,
        memory_units,
        label_text,
        pdims_strategy,
        ideal_line,
        xscale,
    )
