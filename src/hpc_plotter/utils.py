import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def inspect_data(dataframes: Dict[str, pd.DataFrame]):
    """
    Inspect the dataframes.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    """
    print("=" * 80)
    print("Inspecting dataframes...")
    print("=" * 80)
    for method, df in dataframes.items():
        print(f"Method: {method}")
        inspect_df(df)
    print("=" * 80)


def inspect_df(df: pd.DataFrame):
    """
    Inspect the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.
    """
    print(df.to_markdown())
    print("-" * 80)


def plot_with_pdims_strategy(ax: Axes, df: pd.DataFrame, method: str,
                             backend: str, nodes_in_label: bool,
                             pdims_strategy: str, print_decompositions: bool,
                             x_col: str, x_label: str, y_label: str):
    """
    Plot the data based on the pdims strategy.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.
    df : pd.DataFrame
        The dataframe to plot.
    method : str
        The method name.
    backend : str
        The backend name.
    nodes_in_label : bool
        Whether to include node names in labels.
    pdims_strategy : str
        Strategy for plotting pdims ('plot_all' or 'plot_fastest').
    print_decompositions : bool
        Whether to print decompositions on the plot.
    x_col : str
        The column name for the x-axis values.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    """
    if pdims_strategy == 'plot_fastest':
        df_decomp = df.groupby([x_col, 'backend', 'nodes'])
        sorted_dfs = []
        for _, group in df_decomp:
            group.sort_values(by=["time"], inplace=True, ascending=False)
            sorted_dfs.append(group.iloc[0])
        sorted_df = pd.DataFrame(sorted_dfs)
        label = f"{method}-{backend}-{group['nodes'].values[0]}nodes" if nodes_in_label else f"{method}-{backend}"
        ax.plot(sorted_df[x_col].values,
                sorted_df["time"],
                marker='o',
                linestyle='-',
                label=label)
        if print_decompositions:
            for j, (px, py) in enumerate(zip(sorted_df['px'],
                                             sorted_df['py'])):
                ax.annotate(
                    f"{px}x{py}",
                    (sorted_df[x_col].values[j], sorted_df['time'].values[j]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color='red' if j == 0 else 'white')
        return sorted_df[x_col].values, sorted_df["time"].values

    elif pdims_strategy == 'plot_all':
        df_decomp = df.groupby(['decomp'])
        x_values = []
        y_values = []
        for _, group in df_decomp:
            group.drop_duplicates(subset=[x_col, 'decomp'],
                                  keep='last',
                                  inplace=True)
            group.sort_values(by=[x_col], inplace=True, ascending=False)
            label = f"{method}-{backend}-{group['decomp'].values[0]}" if not nodes_in_label else f"{method}-{backend}-{group['nodes'].values[0]}nodes-{group['decomp'].values[0]}"
            ax.plot(group[x_col].values,
                    group["time"],
                    marker='o',
                    linestyle='-',
                    label=label)
            x_values.extend(group[x_col].values)
            y_values.extend(group["time"].values)
        return x_values, y_values


def concatenate_csvs(root_dir: str, output_dir: str):
    """
    Concatenate CSV files and remove duplicates by GPU type.

    Parameters
    ----------
    root_dir : str
        Root directory containing CSV files.
    output_dir : str
        Output directory to save concatenated CSV files.
    """
    # Iterate over each GPU type directory
    for gpu in os.listdir(root_dir):
        gpu_dir = os.path.join(root_dir, gpu)

        # Check if the GPU directory exists and is a directory
        if not os.path.isdir(gpu_dir):
            continue

        # Dictionary to hold combined dataframes for each CSV file name
        combined_dfs = {}

        # List CSV in directory and subdirectories
        for root, dirs, files in os.walk(gpu_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_file_path = os.path.join(root, file)
                    print(f'Concatenating {csv_file_path}...')
                    df = pd.read_csv(csv_file_path,
                                     header=None,
                                     names=[
                                         "rank", "FFT_type", "precision", "x",
                                         "y", "z", "px", "py", "backend",
                                         "nodes", "jit_time", "min_time",
                                         "max_time", "mean_time", "std_time",
                                         "last_time"
                                     ],
                                     index_col=False)
                    if file not in combined_dfs:
                        combined_dfs[file] = df
                    else:
                        combined_dfs[file] = pd.concat(
                            [combined_dfs[file], df], ignore_index=True)

        # Remove duplicates based on specified columns and save
        for file_name, combined_df in combined_dfs.items():
            combined_df.drop_duplicates(subset=[
                "rank", "FFT_type", "precision", "x", "y", "z", "px", "py",
                "backend", "nodes"
            ],
                                        keep='last',
                                        inplace=True)

            gpu_output_dir = os.path.join(output_dir, gpu)
            if not os.path.exists(gpu_output_dir):
                print(f"Creating directory {gpu_output_dir}")
                os.makedirs(gpu_output_dir)

            output_file = os.path.join(gpu_output_dir, file_name)
            print(f"Writing file to {output_file}...")
            combined_df.to_csv(output_file, index=False)


def clean_up_csv(csv_files: List[str],
                 precision: str,
                 fft_type: str,
                 gpus: Optional[List[int]] = None,
                 data_sizes: Optional[List[int]] = None,
                 pdims: Optional[List[str]] = None,
                 pdims_strategy: str = 'plot_fastest',
                 time_aggregation: str = 'mean',
                 backends: List[str] = ['MPI', 'NCCL', 'MPI4JAX'],
                 time_column: str = 'mean_time') -> Dict[str, pd.DataFrame]:
    """
    Clean up and aggregate data from CSV files.

    Parameters
    ----------
    csv_files : List[str]
        List of CSV files to process.
    precision : str
        Precision to filter by.
    fft_type : str
        FFT type to filter by.
    gpus : Optional[List[int]], optional
        List of GPU counts to filter by, by default None.
    data_sizes : Optional[List[int]], optional
        List of data sizes to filter by, by default None.
    pdims : Optional[List[str]], optional
        List of pdims to filter by, by default None.
    pdims_strategy : str, optional
        Strategy for plotting pdims ('plot_all' or 'plot_fastest'), by default 'plot_fastest'.
    time_aggregation : str, optional
        Method of time aggregation ('mean', 'min', 'max'), by default 'mean'.
    time_column : str, optional
        Time column to use for aggregation, by default 'mean_time'.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of method names to aggregated dataframes.
    """
    dataframes = {}
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        df = pd.read_csv(csv_file,
                         header=None,
                         skiprows=1,
                         names=[
                             "rank", "FFT_type", "precision", "x", "y", "z",
                             "px", "py", "backend", "nodes", "jit_time",
                             "min_time", "max_time", "mean_time", "std_div",
                             "last_time"
                         ],
                         dtype={
                             "rank": int,
                             "FFT_type": str,
                             "precision": str,
                             "x": int,
                             "y": int,
                             "z": int,
                             "px": int,
                             "py": int,
                             "backend": str,
                             "nodes": int,
                             "jit_time": float,
                             "min_time": float,
                             "max_time": float,
                             "mean_time": float,
                             "std_div": float,
                             "last_time": float
                         },
                         index_col=False)

        df = df[(df['precision'] == precision) &
                (df['FFT_type'] == fft_type)]  # type : pd.DataFrame
        df = df[df['backend'].isin(backends)]

        if data_sizes:
            df = df[df['x'].isin(data_sizes)]

        if pdims:
            px_list, py_list = zip(*[map(int, p.split('x')) for p in pdims])
            df = df[(df['px'].isin(px_list)) & (df['py'].isin(py_list))]

        grouped_df = df.groupby([
            "FFT_type", "precision", "x", "y", "z", "px", "py", "backend",
            "nodes"
        ])

        sub_dfs = [group for _, group in grouped_df]
        sub_dfs = [
            df.drop_duplicates(subset=[
                "rank", "FFT_type", "precision", "x", "y", "z", "px", "py",
                "backend", "nodes"
            ],
                               keep='last') for df in sub_dfs
        ]

        num_gpu = [len(sub_df) for sub_df in sub_dfs]
        aggregated_dfs = []
        for sub_df in sub_dfs:
            if csv_file.split("/")[-1] == "jaxfft.csv":
                sub_df["time"] = sub_df['max_time']
            elif time_aggregation == 'mean':
                sub_df["time"] = sub_df[time_column].mean()
            elif time_aggregation == 'min':
                sub_df["time"] = sub_df[time_column].min()
            elif time_aggregation == 'max':
                sub_df["time"] = sub_df[time_column].max()

            sub_df.drop(columns=[
                'rank', 'jit_time', 'min_time', 'max_time', 'mean_time',
                'std_div', 'last_time'
            ],
                        inplace=True)
            aggregated_dfs.append(sub_df.iloc[0])

        aggregated_df = pd.DataFrame(aggregated_dfs)
        aggregated_df['gpus'] = num_gpu

        if gpus:
            aggregated_df = aggregated_df[aggregated_df['gpus'].isin(gpus)]

        # After filtering, if we have no data, skip this file
        if aggregated_df.empty:
            continue

        if pdims_strategy == 'plot_all':

            def get_decomp_from_px_py(row):
                if row['px'] > 1 and row['py'] == 1:
                    return 'slab_yz'
                elif row['px'] == 1 and row['py'] > 1:
                    return 'slab_xy'
                else:
                    return 'pencils'

            aggregated_df['decomp'] = aggregated_df.apply(
                get_decomp_from_px_py, axis=1)
            aggregated_df.drop(columns=['px', 'py'], inplace=True)

        if dataframes.get(file_name) is None:
            dataframes[file_name] = aggregated_df
        else:
            dataframes[file_name] = pd.concat(
                [dataframes[file_name], aggregated_df])

    return dataframes
