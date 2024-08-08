Here's the updated README with the additional information about the timer.report and the multi-GPU setup:

# JAX HPC Profiler

JAX HPC Profiler is a tool designed for benchmarking and visualizing performance data in high-performance computing (HPC) environments. It provides functionalities to generate, concatenate, and plot CSV data from various runs.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Generating CSV Files Using the Timer Class](#generating-csv-files-using-the-timer-class)
- [CSV Structure](#csv-structure)
- [Concatenating Files from Different Runs](#concatenating-files-from-different-runs)
- [Plotting CSV Data](#plotting-csv-data)
- [Examples](#examples)

## Introduction
JAX HPC Profiler allows users to:
1. Generate CSV files containing performance data.
2. Concatenate multiple CSV files from different runs.
3. Plot the performance data for analysis.

## Installation

To install the package, run the following command:

```bash
pip install jax-hpc-profiler
```

## Generating CSV Files Using the Timer Class

To generate CSV files, you can use the `Timer` class provided in the `jax_hpc_profiler.timer` module. This class helps in timing functions and saving the timing results to CSV files.

### Example Usage

```python
import jax
from jax_hpc_profiler import Timer

def fcn(m, n, k):
    return jax.numpy.dot(m, n) + k

timer = Timer(save_jaxpr=True)
m = jax.numpy.ones((1000, 1000))
n = jax.numpy.ones((1000, 1000))
k = jax.numpy.ones((1000, 1000))

timer.chrono_jit(fcn, m, n, k)
for i in range(10):
    timer.chrono_fun(fcn, m, n, k)

meta_data = {
  "function": "fcn",
  "precision": "float32",
  "x": 1000,
  "y": 1000,
  "z": 1000,
  "px": 1,
  "py": 1,
  "backend": "NCCL",
  "nodes": 1
}
extra_info = {
    "done": "yes"
}

timer.report("examples/profiling/test.csv", **meta_data,  extra_info=extra_info)
```

`timer.report` has sensible defaults and this is the API for the `Timer` class:

- `csv_filename`: The path to the CSV file to save the timing data **(required)**.
- `function`: The name of the function being timed **(required)**.
- `x`: The size of the input data in the x dimension **(required)**.
- `y`: The size of the input data in the y dimension (by default same as x).
- `z`: The size of the input data in the z dimension (by default same as x).
- `precision`: The precision of the data (default: "float32").
- `px`: The number of partitions in the x dimension (default: 1).
- `py`: The number of partitions in the y dimension (default: 1).
- `backend`: The backend used for computation (default: "NCCL").
- `nodes`: The number of nodes used for computation (default: 1).
- `md_filename`: The path to the markdown file containing the compiled code and other information (default: {csv_folder}/{x}_{px}_{py}_{backend}_{precision}_{function}.md).
- `extra_info`: Additional information to include in the report (default: {}

`px` and `py` are used to specify the data decomposition. For example, if you have a 2D array of size 1000x1000 and you partition it into 4 parts (2x2), you would set `px=2` and `py=2`.\
they can also be used in a single device run to specify batch size.

Some decomposition parameters are generated and that are specific to 3D data decomposition.\
`slab_yz` if the distributed axis is the y-axis.\
`slab_xy` if the distributed axis is the x-axis.\
`pencils` if the distributed axis are the x and y axes.

### Multi-GPU Setup

In a multi-GPU setup, the times are automatically averaged across ranks, providing a single performance metric for the entire setup.

## CSV Structure

The CSV files should follow a specific structure to ensure proper processing and concatenation. The directory structure should be organized by GPU type, with subdirectories for the number of GPUs and the respective CSV files.

### Example Directory Structure

```
root_directory/
├── gpu_1/
│   ├── 2/
│   │   ├── method_1.csv
│   │   ├── method_2.csv
│   │   └── method_3.csv
│   ├── 4/
│   │   ├── method_1.csv
│   │   ├── method_2.csv
│   │   └── method_3.csv
│   └── 8/
│       ├── method_1.csv
│       ├── method_2.csv
│       └── method_3.csv
└── gpu_2/
    ├── 2/
    │   ├── method_1.csv
    │   ├── method_2.csv
    │   └── method_3.csv
    ├── 4/
    │   ├── method_1.csv
    │   ├── method_2.csv
    │   └── method_3.csv
    └── 8/
        ├── method_1.csv
        ├── method_2.csv
        └── method_3.csv
```

## Concatenating Files from Different Runs

The `plot` function expects the directory to be organized as described above, but with the different number of GPUs together in the same directory. The `concatenate` function can be used to concatenate the CSV files from different runs into a single file.

### Example Usage

```bash
jax-hpc-profiler concat /path/to/root_directory /path/to/output
```

And the output will be:

```
out_directory/
├── gpu_1/
│   ├── method_1.csv
│   ├── method_2.csv
│   └── method_3.csv
└── gpu_2/
    ├── method_1.csv
    ├── method_2.csv
    └── method_3.csv
```

## Plotting CSV Data

You can plot the performance data using the `plot` command. The plotting command provides various options to customize the plots.

### Usage

```bash
jax-hpc-profiler plot -f <csv_files> [options]
```

### Options

- `-f, --csv_files`: List of CSV files to plot (required).
- `-g, --gpus`: List of number of GPUs to plot.
- `-d, --data_size`: List of data sizes to plot.
- `-fd, --filter_pdims`: List of pdims to filter (e.g., 1x4 2x2 4x8).
- `-ps, --pdim_strategy`: Strategy for plotting pdims. This argument can be multiple ones (`plot_all`, `plot_fastest`, `slab_yz`, `slab_xy`, `pencils`).
  - `plot_all`: Plot every decomposition.
  - `plot_fastest`: Plot the fastest decomposition.
- `-pr, --precision`: Precision to filter by. This argument can be multiple ones (`float32`, `float64`).
- `-fn, --function_name`: Function names to filter. This argument can be multiple ones.
- `-pt, --plot_times`: Time columns to plot (`jit_time`, `min_time`, `max_time`, `mean_time`, `std_time`, `last_time`). Note: You cannot plot memory and time together.
- `-pm, --plot_memory`: Memory columns to plot (`generated_code`, `argument_size`, `output_size`, `temp_size`). Note: You cannot plot memory and time together.
- `-mu, --memory_units`: Memory units to plot (`KB`, `MB`, `GB`, `TB`).
- `-fs, --figure_size`: Figure size.
- `-o, --output`: Output file (if none then only show plot).
- `-db, --dark_bg`: Use dark background for plotting.
- `-pd, --print_decompositions`: Print decompositions on plot (experimental).
- `-b, --backends`: List of backends to include. This argument can be multiple ones.
- `-sc, --scaling`: Scaling type (`Weak`, `Strong`).
- `-l, --label_text`: Custom label for the plot. You can use placeholders: `%decomposition%` (or `%p%`), `%precision%` (or `%pr%`), `%plot_name%` (or `%pn%`), `%backend%` (or `%b%`), `%node%` (or `%n%`), `%methodname%` (or `%m%`).

## Examples

The repository includes examples for both profiling and plotting.

### Profiling Example

See the `examples/profiling` directory for profiling examples, including `function.py`, `test.csv`, and the generated markdown report.

### Plotting Example

See the `examples/plotting` directory for plotting examples, including `generator.py`, `sample_data1.csv`, `sample_data2.csv`, and `sample_data3.csv`.

a multi GPU example comparing distributed FFT can be found here [jaxdecomp-bechmarks](https://github.com/ASKabalan/jaxdecomp-benchmarks)
