import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
from matplotlib.axes import Axes

COLUMN_NAMES_19 = [
    'function',
    'precision',
    'x',
    'y',
    'z',
    'px',
    'py',
    'backend',
    'nodes',
    'jit_time',
    'min_time',
    'max_time',
    'mean_time',
    'std_div',
    'last_time',
    'generated_code',
    'argument_size',
    'output_size',
    'temp_size',
]

COLUMN_NAMES_20 = COLUMN_NAMES_19 + ['flops']

COLUMN_DTYPES = {
    'function': str,
    'precision': str,
    'x': int,
    'y': int,
    'z': int,
    'px': int,
    'py': int,
    'backend': str,
    'nodes': int,
    'jit_time': float,
    'min_time': float,
    'max_time': float,
    'mean_time': float,
    'std_div': float,
    'last_time': float,
    'generated_code': float,
    'argument_size': float,
    'output_size': float,
    'temp_size': float,
    'flops': float,
}


def _read_csv(csv_file: str) -> pd.DataFrame:
    """Read headerless CSV, auto-detecting 19 or 20 columns."""
    with open(csv_file) as f:
        first_line = f.readline()
    num_fields = len(first_line.strip().split(','))

    if num_fields >= 20:
        names = COLUMN_NAMES_20
    else:
        names = COLUMN_NAMES_19

    dtypes = {k: v for k, v in COLUMN_DTYPES.items() if k in names}

    return pd.read_csv(
        csv_file,
        header=None,
        skiprows=0,
        names=names,
        dtype=dtypes,
        index_col=False,
    )


def inspect_data(dataframes: Dict[str, pd.DataFrame]) -> None:
    """
    Inspect the dataframes.

    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of method names to dataframes.
    """
    print('=' * 80)
    print('Inspecting dataframes...')
    print('=' * 80)
    for method, df in dataframes.items():
        print(f'Method: {method}')
        inspect_df(df)
    print('=' * 80)


def inspect_df(df: pd.DataFrame) -> None:
    """
    Inspect the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.
    """
    print(df.to_markdown())
    print('-' * 80)


params_dict = {
    '%pn%': '%plot_name%',
    '%m%': '%method_name%',
    '%n%': '%node%',
    '%b%': '%backend%',
    '%f%': '%function%',
    '%cn%': '%column_name%',
    '%pr%': '%precision%',
    '%p%': '%decomposition%',
    '%d%': '%data_size%',
    '%g%': '%nb_gpu%',
}


def expand_label(label_template: str, params: dict[str, str]) -> str:
    """
    Expand the label template with the provided parameters.

    Parameters
    ----------
    label_template : str
        The label template with placeholders.
    params : dict
        The dictionary with actual values to replace placeholders.

    Returns
    -------
    str
        The expanded label.
    """
    for key, value in params_dict.items():
        label_template = label_template.replace(key, value)

    for key, value in params.items():
        label_template = label_template.replace(f'%{key}%', str(value))
    return label_template


def plot_with_pdims_strategy(
    ax: Axes,
    df: pd.DataFrame,
    method: str,
    pdims_strategy: List[str],
    print_decompositions: bool,
    x_col: str,
    y_col: str,
    label_template: str,
) -> Optional[Tuple[List[float], List[float]]]:
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
    pdims_strategy : List[str]
        Strategy for plotting pdims.
    print_decompositions : bool
        Whether to print decompositions on the plot.
    x_col : str
        The column name for the x-axis values.
    y_col : str
        The column name for the y-axis values.
    label_template : str
        Template for plot labels with placeholders.
    """
    label_params = {
        'plot_name': y_col,
        'method_name': method,
        'backend': df['backend'].values[0],
        'node': df['nodes'].values[0],
        'precision': df['precision'].values[0],
        'function': df['function'].values[0],
    }

    if 'plot_fastest' in pdims_strategy:
        df_decomp = df.groupby([x_col])

        # Sort all and keep fastest
        sorted_dfs = []
        for _, group in df_decomp:
            group.sort_values(by=[y_col], inplace=True, ascending=True)
            sorted_dfs.append(group.iloc[0])
        sorted_df = pd.DataFrame(sorted_dfs)
        label_params.update({'decomposition': f'{group["px"].values[0]}x{group["py"].values[0]}'})
        label = expand_label(label_template, label_params)
        ax.plot(
            sorted_df[x_col].values,
            sorted_df[y_col],
            marker='o',
            linestyle='-',
            label=label,
        )
        # TODO(wassim) : this is not working very well
        if print_decompositions:
            for j, (px, py) in enumerate(zip(sorted_df['px'], sorted_df['py'])):
                ax.annotate(
                    f'{px}x{py}',
                    (sorted_df[x_col].values[j], sorted_df[y_col].values[j]),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha='center',
                    color='red' if j == 0 else 'white',
                )
        return sorted_df[x_col].values, sorted_df[y_col].values

    elif any(
        strategy in pdims_strategy for strategy in ['plot_all', 'slab_yz', 'slab_xy', 'pencils']
    ):
        df_decomp = df.groupby(['decomp'])
        x_values = []
        y_values = []
        for _, group in df_decomp:
            group.drop_duplicates(subset=[x_col, 'decomp'], keep='last', inplace=True)
            group.sort_values(by=[x_col], inplace=True, ascending=False)
            # filter decomp based on pdims_strategy
            if 'plot_all' not in pdims_strategy and group['decomp'].values[0] not in pdims_strategy:
                continue

            label_params.update({'decomposition': group['decomp'].values[0]})
            label = expand_label(label_template, label_params)
            ax.plot(
                group[x_col].values,
                group[y_col],
                marker='o',
                linestyle='-',
                label=label,
            )
            x_values.extend(group[x_col].values)
            y_values.extend(group[y_col].values)
        return x_values, y_values


def _parse_volume_query(query: str) -> tuple[str, int]:
    """Parse 'global_N', 'global_NxMxK', 'local_N', 'local_NxMxK' into (column, volume).

    Parameters
    ----------
    query : str
        Volume query string.

    Returns
    -------
    tuple[str, int]
        (column_name, volume_int).

    Raises
    ------
    ValueError
        If *query* doesn't start with ``global_`` or ``local_``.
    """
    for prefix, col in [('global_', 'global_vol'), ('local_', 'local_vol')]:
        if query.startswith(prefix):
            value_part = query[len(prefix):]
            if 'x' in value_part:
                vol = 1
                for dim in value_part.split('x'):
                    vol *= int(dim)
            else:
                vol = int(value_part)
            return col, vol
    raise ValueError(f'Query must start with global_ or local_: {query!r}')


def _query_volume_type(query: str) -> str:
    """Return 'global_vol' or 'local_vol' for a volume query."""
    col, _ = _parse_volume_query(query)
    return col


def parse_data_size_grep(df: pd.DataFrame, query: str) -> pd.Series:
    """
    Parse a data-size query and return a boolean mask over the DataFrame.

    Requires ``global_vol`` and ``local_vol`` columns on *df*.

    Supported query formats:

    | Query                  | Matches                                    |
    |------------------------|--------------------------------------------|
    | ``global_N``           | ``global_vol == N``  (exact volume)        |
    | ``global_NxMxK``       | ``global_vol == N*M*K``                    |
    | ``local_N``            | ``local_vol == N``   (exact volume)        |
    | ``local_NxMxK``        | ``local_vol == N*M*K``                     |

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``global_vol``, ``local_vol``.
    query : str
        The filter query string.

    Returns
    -------
    pd.Series
        Boolean mask.

    Raises
    ------
    ValueError
        If *query* does not match any recognized format.
    """
    col, vol = _parse_volume_query(query.strip())
    return df[col] == vol


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
                    df = _read_csv(csv_file_path)
                    if file not in combined_dfs:
                        combined_dfs[file] = df
                    else:
                        combined_dfs[file] = pd.concat([combined_dfs[file], df], ignore_index=True)

        # Remove duplicates based on specified columns and save
        for file_name, combined_df in combined_dfs.items():
            combined_df.drop_duplicates(
                subset=[
                    'function',
                    'precision',
                    'x',
                    'y',
                    'z',
                    'px',
                    'py',
                    'backend',
                    'nodes',
                ],
                keep='last',
                inplace=True,
            )

            gpu_output_dir = os.path.join(output_dir, gpu)
            if not os.path.exists(gpu_output_dir):
                print(f'Creating directory {gpu_output_dir}')
                os.makedirs(gpu_output_dir)

            output_file = os.path.join(gpu_output_dir, file_name)
            print(f'Writing file to {output_file}...')
            combined_df.to_csv(output_file, index=False)


def clean_up_csv(
    csv_files: List[str],
    precisions: Optional[List[str]] = None,
    function_names: Optional[List[str]] = None,
    gpus: Optional[List[int]] = None,
    data_size_queries: Optional[List[str]] = None,
    pdims: Optional[List[str]] = None,
    pdims_strategy: List[str] = ['plot_fastest'],
    backends: Optional[List[str]] = None,
    memory_units: str = 'KB',
) -> Tuple[Dict[str, pd.DataFrame], List[int], List[int]]:
    """
    Clean up and aggregate data from CSV files.

    Parameters
    ----------
    csv_files : List[str]
        List of CSV files to process.
    precisions : Optional[List[str]], optional
        Precisions to filter by, by default None.
    function_names : Optional[List[str]], optional
        Function names to filter by, by default None.
    gpus : Optional[List[int]], optional
        List of GPU counts to filter by, by default None.
    data_size_queries : Optional[List[str]], optional
        List of data size queries (grep-like) to filter by, by default None.
    pdims : Optional[List[str]], optional
        List of pdims to filter by, by default None.
    pdims_strategy : List[str], optional
        Strategy for plotting pdims, by default ['plot_fastest'].
    backends : List[str], optional
        List of backends to filter by, by default None.
    memory_units : str, optional
        Memory unit for conversion, by default 'KB'.

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], List[int], List[int]]
        Dictionary of method names to aggregated dataframes,
        available GPU counts, and available global volumes.
    """
    dataframes = {}
    available_gpu_counts = set()
    available_global_vols = set()
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        ext = os.path.splitext(os.path.basename(csv_file))[1]
        if ext != '.csv':
            print(f'Ignoring {csv_file} as it is not a CSV file')
            continue

        df = _read_csv(csv_file)

        # Filter precisions
        if precisions:
            df = df[df['precision'].isin(precisions)]
        # Filter function names
        if function_names:
            df = df[df['function'].isin(function_names)]
        # Filter backends
        if backends:
            df = df[df['backend'].isin(backends)]

        # convert memory units columns to requested memory_units
        match memory_units:
            case 'KB':
                factor = 1024
            case 'MB':
                factor = 1024**2
            case 'GB':
                factor = 1024**3
            case 'TB':
                factor = 1024**4
            case _:
                factor = 1

        df['generated_code'] = df['generated_code'] / factor
        df['argument_size'] = df['argument_size'] / factor
        df['output_size'] = df['output_size'] / factor
        df['temp_size'] = df['temp_size'] / factor
        # in case of the same test is run multiple times, keep the last one
        df = df.drop_duplicates(
            subset=[
                'function',
                'precision',
                'x',
                'y',
                'z',
                'px',
                'py',
                'backend',
                'nodes',
            ],
            keep='last',
        )

        df['gpus'] = df['px'] * df['py']
        df['global_vol'] = df['x'] * df['y'] * df['z']
        df['local_vol'] = df['global_vol'] // df['gpus']

        if gpus:
            df = df[df['gpus'].isin(gpus)]

        # Filter data sizes using grep-like queries
        if data_size_queries:
            mask = pd.Series(False, index=df.index)
            for q in data_size_queries:
                mask |= parse_data_size_grep(df, q)
            df = df[mask]

        # Filter pdims
        if pdims:
            px_list, py_list = zip(*[map(int, p.split('x')) for p in pdims])
            df = df[(df['px'].isin(px_list)) & (df['py'].isin(py_list))]

        if (
            'plot_all' in pdims_strategy
            or 'slab_yz' in pdims_strategy
            or 'slab_xy' in pdims_strategy
            or 'pencils' in pdims_strategy
        ):

            def get_decomp_from_px_py(row):
                if row['px'] > 1 and row['py'] == 1:
                    return 'slab_yz'
                elif row['px'] == 1 and row['py'] > 1:
                    return 'slab_xy'
                else:
                    return 'pencils'

            df['decomp'] = df.apply(get_decomp_from_px_py, axis=1)
            df.drop(columns=['px', 'py'], inplace=True)
            if 'plot_all' not in pdims_strategy:
                df = df[df['decomp'].isin(pdims_strategy)]

        # check available gpus and global volumes in dataset
        available_gpu_counts.update(df['gpus'].unique())
        available_global_vols.update(df['global_vol'].unique())

        if dataframes.get(file_name) is None:
            dataframes[file_name] = df
        else:
            dataframes[file_name] = pd.concat([dataframes[file_name], df])

    print(f'requested GPUS: {gpus} available GPUS: {available_gpu_counts}')
    print(
        f'requested data sizes: {data_size_queries} available global volumes: {available_global_vols}'
    )

    available_gpu_counts = (
        available_gpu_counts
        if gpus is None
        else [gpu for gpu in gpus if gpu in available_gpu_counts]
    )
    available_global_vols = sorted(available_global_vols)

    return dataframes, available_gpu_counts, available_global_vols


def _format_volume(vol: int) -> str:
    """Format a volume as N³ if it is a perfect cube, otherwise as the raw number."""
    cbrt = round(vol ** (1.0 / 3.0))
    if cbrt**3 == vol:
        return f'{cbrt}\u00b3'
    return f'{vol:,}'


def probe_csv_metadata(csv_files: List[str]) -> str:
    """
    Load CSVs and return a formatted report of available metadata.

    Parameters
    ----------
    csv_files : List[str]
        List of CSV file paths.

    Returns
    -------
    str
        Formatted probe report.
    """
    frames = []
    for csv_file in csv_files:
        ext = os.path.splitext(csv_file)[1]
        if ext != '.csv':
            continue
        frames.append(_read_csv(csv_file))

    if not frames:
        return 'No CSV files found.'

    df = pd.concat(frames, ignore_index=True)
    df['gpus'] = df['px'] * df['py']
    df['global_vol'] = df['x'] * df['y'] * df['z']
    df['local_vol'] = df['global_vol'] // df['gpus']

    lines = []
    lines.append('=== CSV Probe Report ===')
    lines.append('')
    lines.append(f'Files: {", ".join(os.path.basename(f) for f in csv_files)}')
    lines.append('')
    lines.append(f'Functions:  {", ".join(sorted(df["function"].unique()))}')
    lines.append(f'Backends:   {", ".join(sorted(df["backend"].unique()))}')
    lines.append(f'Precisions: {", ".join(sorted(df["precision"].unique()))}')
    gpu_counts = sorted(df['gpus'].unique())
    lines.append(f'GPU counts: {", ".join(str(g) for g in gpu_counts)}')
    lines.append('')

    # Table header
    lines.append(
        f'{"Query Alias":<24} {"Global Volume":<16} {"Local Volume":<16} '
        f'{"GPUs Available":<20} {"Shapes (X×Y×Z)":<20}'
    )
    lines.append('-' * 96)

    dash = '\u2014'

    # Global volumes
    for gv in sorted(df['global_vol'].unique()):
        subset = df[df['global_vol'] == gv]
        gpus_avail = sorted(subset['gpus'].unique())
        shapes = subset[['x', 'y', 'z']].drop_duplicates()
        shape_strs = ', '.join(f'{r.x}\u00d7{r.y}\u00d7{r.z}' for r in shapes.itertuples())

        alias = f'global_{gv}'
        vol_str = _format_volume(gv)
        gpus_str = ', '.join(str(g) for g in gpus_avail)

        lines.append(
            f'{alias:<24} {vol_str:<16} {dash:<16} {gpus_str:<20} {shape_strs:<20}'
        )

    # Local volumes
    for lv in sorted(df['local_vol'].unique()):
        subset = df[df['local_vol'] == lv]
        gpus_avail = sorted(subset['gpus'].unique())
        shapes = subset[['x', 'y', 'z']].drop_duplicates()
        shape_strs = ', '.join(f'{r.x}\u00d7{r.y}\u00d7{r.z}' for r in shapes.itertuples())

        alias = f'local_{lv}'
        vol_str = _format_volume(lv)
        gpus_str = ', '.join(str(g) for g in gpus_avail)

        lines.append(
            f'{alias:<24} {dash:<16} {vol_str:<16} {gpus_str:<20} {shape_strs:<20}'
        )

    return '\n'.join(lines)
