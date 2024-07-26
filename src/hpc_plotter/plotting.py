from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

from .utils import plot_with_pdims_strategy

np.seterr(divide='ignore')
plt.rcParams.update({'font.size': 10})


def configure_axes(ax: Axes, x_values: List[int], y_values: List[float],
                   xlabel: str, ylabel: str):
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
    ylabel : str
        The label for the y-axis.
    """
    f2 = lambda x: np.log2(x)
    g2 = lambda x: 2**x
    ax.set_xlim([min(x_values), max(x_values)])
    y_min, y_max = min(y_values) * 0.9, max(y_values) * 1.1
    ax.set_ylim([y_min, y_max])
    ax.set_xscale('function', functions=(f2, g2))
    ax.set_yscale('symlog')
    ax.set_xticks(x_values)
    time_ticks = [
        10**t for t in range(int(np.floor(np.log10(y_min))), 1 +
                             int(np.ceil(np.log10(y_max))))
    ]
    ax.set_yticks(time_ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for x_value in x_values:
        ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='lower center',
              bbox_to_anchor=(0.5, 0.05),
              ncol=4,
              prop={'size': 14})


def plot_strong_scaling(dataframes: Dict[str, pd.DataFrame],
                        fixed_data_size: List[int],
                        nodes_in_label: bool = False,
                        figure_size: tuple = (6, 4),
                        output: Optional[str] = None,
                        dark_bg: bool = False,
                        print_decompositions: bool = False,
                        backends: Optional[List[str]] = None,
                        pdims_strategy: str = 'plot_fastest'):
    """
    Plot strong scaling based on the number of GPUs.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    fixed_data_size : List[int]
        List of fixed data sizes to plot.
    nodes_in_label : bool, optional
        Whether to include node names in labels, by default False.
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

    if backends is None:
        backends = ['MPI', 'NCCL', 'MPI4JAX', 'NCCL_PL', 'MPI_P2P']

    num_subplots = len(fixed_data_size)
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, data_size in enumerate(fixed_data_size):
        ax: Axes = axs[i]
        number_of_gpus = []
        times = []

        for method, df in dataframes.items():
            df = df[df['x'] == int(data_size)]
            if df.empty:
                continue
            df = df.sort_values(by=['gpus'])
            number_of_gpus.extend(df['gpus'].values)
            times.extend(df["time"].values)

            for backend in backends:
                df_backend = df[df['backend'] == backend]
                if df_backend.empty:
                    continue

                x_values, y_values = plot_with_pdims_strategy(
                    ax, df_backend, method, backend, nodes_in_label,
                    pdims_strategy, print_decompositions, 'gpus',
                    'Number of GPUs', 'Time (milliseconds)')
                number_of_gpus.extend(x_values)
                times.extend(y_values)

        configure_axes(ax, number_of_gpus, times, 'Number of GPUs',
                       'Time (milliseconds)')

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
        plt.savefig(output, bbox_inches='tight', transparent=False)


def plot_weak_scaling(dataframes: Dict[str, pd.DataFrame],
                      fixed_gpu_size: List[int],
                      nodes_in_label: bool = False,
                      figure_size: tuple = (6, 4),
                      output: Optional[str] = None,
                      dark_bg: bool = False,
                      print_decompositions: bool = False,
                      backends: Optional[List[str]] = None,
                      pdims_strategy: str = 'plot_fastest'):
    """
    Plot weak scaling based on the data size.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    fixed_gpu_size : List[int]
        List of fixed GPU sizes to plot.
    nodes_in_label : bool, optional
        Whether to include node names in labels, by default False.
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

    if backends is None:
        backends = ['MPI', 'NCCL', 'MPI4JAX', 'NCCL_PL', 'MPI_P2P']

    num_subplots = len(fixed_gpu_size)
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, gpu_size in enumerate(fixed_gpu_size):
        ax: Axes = axs[i]
        data_sizes = []
        times = []

        for method, df in dataframes.items():
            df = df[df['gpus'] == int(gpu_size)]
            if df.empty:
                continue

            df = df.sort_values(by=['x'])
            data_sizes.extend(df['x'].values)
            times.extend(df["time"].values.astype(np.float32))

            for backend in backends:
                df_backend = df[df['backend'] == backend]
                if df_backend.empty:
                    continue

                x_values, y_values = plot_with_pdims_strategy(
                    ax, df_backend, method, backend, nodes_in_label,
                    pdims_strategy, print_decompositions, 'x',
                    'Data size (pixels³)', 'Time (seconds)')
                data_sizes.extend(x_values)
                times.extend(y_values)

        configure_axes(ax, data_sizes, times, 'Data size (pixels³)',
                       'Time (seconds)')

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
        plt.savefig(output, bbox_inches='tight', transparent=False, dpi=300)
