# JAX HPC Profiler

[![Build](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/python-publish.yml)
[![Code Formatting](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/formatting.yml/badge.svg)](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/formatting.yml)
[![Tests](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/tests.yml/badge.svg)](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/tests.yml)
[![Notebooks](https://img.shields.io/github/actions/workflow/status/ASKabalan/jax-hpc-profiler/notebooks.yml?logo=jupyter&label=notebooks)](https://github.com/ASKabalan/jax-hpc-profiler/actions/workflows/notebooks.yml)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)

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
jhp concat /path/to/root_directory /path/to/output
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

## Inspecting CSV Metadata

You can inspect available metadata in your CSV files using the `probe` command:

```bash
jhp probe -f <csv_files>
```

This prints the available data sizes, GPU counts, functions, backends, precisions, and other metadata found in the CSV files. Use this to understand what filters to apply before plotting.

## Plotting CSV Data

You can plot the performance data using the `plot` command. The plotting command provides various options to customize the plots.

### Usage

```bash
jhp plot -f <csv_files> [options]
```

### Options

- `-f, --csv_files`: List of CSV files to plot (required).
- `-sc, --scaling`: Axis mode (required):
  - `data` (or `d`): subplots per data size, x-axis = GPUs (strong scaling view).
  - `GPUs` (or `g`): subplots per GPU count, x-axis = data size.
- `-g, --gpus`: List of GPU counts to filter.
- `-d, --data_size`: Data size queries. Examples: `global_2097152`, `global_128x128x128`, `local_2097152`, `local_128x128x128`. Bare integers are auto-translated to `global_NxNxN` (cubed).
- `-fd, --filter_pdims`: List of pdims to filter (e.g., 1x4 2x2 4x8).
- `-ps, --pdim_strategy`: Strategy for plotting pdims (`plot_all`, `plot_fastest`, `slab_yz`, `slab_xy`, `pencils`).
- `-pr, --precision`: Precision to filter by (`float32`, `float64`).
- `-fn, --function_name`: Function names to filter.
- `-pt, --plot_times`: Time columns to plot (`jit_time`, `min_time`, `max_time`, `mean_time`, `std_time`, `last_time`). Note: You cannot plot memory and time together.
- `-pm, --plot_memory`: Memory columns to plot (`generated_code`, `argument_size`, `output_size`, `temp_size`). Note: You cannot plot memory and time together.
- `-mu, --memory_units`: Memory units to plot (`KB`, `MB`, `GB`, `TB`).
- `-fs, --figure_size`: Figure size.
- `-o, --output`: Output file (if none then only show plot).
- `-pd, --print_decompositions`: Print decompositions on plot (experimental).
- `-b, --backends`: List of backends to include.
- `--ideal_line`: Overlay an ideal scaling reference line (1/N for global data sizes, flat for local data sizes).
- `-xs, --xscale`: X-axis scale (`linear`, `symlog`, `log2`, `log10`).
- `-xl, --xlabel`: Custom x-axis label.
- `-tl, --title`: Custom plot title.
- `-l, --label_text`: Custom label for the plot. You can use placeholders: `%decomposition%` (or `%p%`), `%precision%` (or `%pr%`), `%plot_name%` (or `%pn%`), `%backend%` (or `%b%`), `%node%` (or `%n%`), `%methodname%` (or `%m%`), `%function%` (or `%f%`).

### CLI examples

**Strong scaling** (subplots per data size, x-axis = GPUs):

```bash
jhp plot -f DATA.csv -sc data -d 128 256 512 -pt mean_time --ideal_line
```

**Size scaling** (subplots per GPU count, x-axis = data size):

```bash
jhp plot -f DATA.csv -sc GPUs -pt mean_time
```

## Examples

The repository includes Jupyter notebook examples:

- **`examples/profiling.ipynb`**: Single-device profiling of JAX and NumPy functions with `Timer`, CSV report generation, and plotting with `plot_by_gpus`.
- **`examples/distributed_profiling.ipynb`**: Multi-device profiling with sharded arrays, `plot_by_gpus`, `plot_by_data_size`, `probe_csv_metadata`, and CLI usage.

A multi-GPU example comparing distributed FFT can be found here: [jaxdecomp-benchmarks](https://github.com/ASKabalan/jaxdecomp-benchmarks)
