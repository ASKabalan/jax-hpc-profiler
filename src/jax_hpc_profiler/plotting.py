from itertools import product
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from .utils import clean_up_csv, inspect_df, plot_with_pdims_strategy

np.seterr(divide="ignore")


def configure_axes(
    ax: Axes,
    x_values: List[int],
    y_values: List[float],
    xlabel: str,
    title: str,
    plotting_memory: bool = False,
    memory_units: str = "bytes",
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
    ylabel = ("Time (milliseconds)"
              if not plotting_memory else f"Memory ({memory_units})")
    f2 = lambda x: np.log2(x)
    g2 = lambda x: 2**x
    ax.set_xlim([min(x_values), max(x_values)])
    y_min, y_max = min(y_values) * 0.6, max(y_values) * 1.1
    ax.set_title(title)
    ax.set_ylim([y_min, y_max])
    ax.set_xscale("function", functions=(f2, g2))
    if not plotting_memory:
        ax.set_yscale("symlog")
        time_ticks = [
            10**t for t in range(int(np.floor(np.log10(y_min))), 1 +
                                 int(np.ceil(np.log10(y_max))))
        ]
        ax.set_yticks(time_ticks)
    ax.set_xticks(x_values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for x_value in x_values:
        ax.axvline(x=x_value, color="gray", linestyle="--", alpha=0.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        fontsize="x-large",
        prop={"size": 14},
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
    plot_columns: List[str] = ["mean_time"],
    memory_units: str = "bytes",
    label_text: str = "plot",
    pdims_strategy: List[str] = ["plot_fastest"],
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
        plt.style.use("dark_background")

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
            filtered_method_df = filtered_method_df.sort_values(
                by=[size_column])
            functions = (pd.unique(filtered_method_df["function"])
                         if functions is None else functions)
            precisions = (pd.unique(filtered_method_df["precision"])
                          if precisions is None else precisions)
            backends = (pd.unique(filtered_method_df["backend"])
                        if backends is None else backends)

            combinations = product(backends, precisions, functions,
                                   plot_columns)

            for backend, precision, function, plot_column in combinations:

                filtered_params_df = filtered_method_df[
                    (filtered_method_df["backend"] == backend)
                    & (filtered_method_df["precision"] == precision)
                    & (filtered_method_df["function"] == function)]
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
            plotting_memory = "time" not in plot_columns[0].lower()
            configure_axes(
                ax,
                x_values,
                y_values,
                f"{title} {fixed_size}",
                xlabel,
                plotting_memory,
                memory_units,
            )

    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    fig.tight_layout()
    rect = FancyBboxPatch((0.1, 0.1),
                          0.8,
                          0.8,
                          boxstyle="round,pad=0.02",
                          ec="black",
                          fc="none")
    fig.patches.append(rect)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight", transparent=True)


def plot_strong_scaling(
    csv_files: List[str],
    fixed_gpu_size: Optional[List[int]] = None,
    fixed_data_size: Optional[List[int]] = None,
    functions: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ["plot_fastest"],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ["mean_time"],
    memory_units: str = "bytes",
    label_text: str = "%m%-%f%-%pn%-%pr%-%b%-%p%-%n%",
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
        print(f"No dataframes found for the given arguments. Exiting...")
        return

    plot_scaling(
        dataframes,
        available_data_sizes,
        "gpus",
        "x",
        "Number of GPUs",
        "Data size",
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
    pdims_strategy: List[str] = ["plot_fastest"],
    print_decompositions: bool = False,
    backends: Optional[List[str]] = None,
    plot_columns: List[str] = ["mean_time"],
    memory_units: str = "bytes",
    label_text: str = "%m%-%f%-%pn%-%pr%-%b%-%p%-%n%",
    figure_size: tuple = (6, 4),
    dark_bg: bool = False,
    output: Optional[str] = None,
):
    """
    Plot weak scaling based on the data size.
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
        print(f"No dataframes found for the given arguments. Exiting...")
        return

    plot_scaling(
        dataframes,
        available_gpu_counts,
        "x",
        "gpus",
        "Data size",
        "Number of GPUs",
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
