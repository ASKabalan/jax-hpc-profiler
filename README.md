# HPC Plotter

HPC Plotter is a tool designed for benchmarking and visualizing performance data in high-performance computing (HPC) environments. It provides functionalities to generate, concatenate, and plot CSV data from various runs.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Generating CSV Files Using the Timer Class](#generating-csv-files-using-the-timer-class)
- [CSV Structure](#csv-structure)
- [Concatenating Files from Different Runs](#concatenating-files-from-different-runs)
- [Plotting CSV Data](#plotting-csv-data)

## Introduction
HPC Plotter allows users to:
1. Generate CSV files containing performance data.
2. Concatenate multiple CSV files from different runs.
3. Plot the performance data for analysis.

## Installation

To install the package, run the following command:

```bash
pip install hpc-plotter
```
## Generating CSV Files Using the Timer Class

To generate CSV files, you can use the `Timer` class provided in the `hpc_plotter.timer` module. This class helps in timing functions and saving the timing results to CSV files.

### Example Usage

```python
import time
from hpc_plotter.timer import Timer
import jax
# Define the functions you want to time
def example_function():
    time.sleep(1)  # Simulating a task

# Create a Timer instance
timer = Timer()

# Time the function
timer.chrono_jit(example_function)
for _ in range(5):
    timer.chrono_fun(example_function)

# Metadata for the CSV file
metadata = {
    'rank': jax.process_index(),
    'function_name': 'example_function',
    'precision': 'float32',
    'x': '1024',
    'y': '1024',
    'z': '1024',
    'px': '4',
    'py': '4',
    'backend': 'NCCL',
    'nodes': '2'
}

# Print the results to a CSV file
timer.print_to_csv('output.csv', **metadata)
```

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

The `plot` function expects the directory to be organized as described above, but with the different number of GPUs toghether in the same directory. The `concatenate` function can be used to concatenate the CSV files from different runs into a single file.

### Example Usage

```bash
hpc-plotter concat /path/to/root_directory /path/to/output
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
hpc-plotter plot -f <csv_files> [options]
```

with options :


- `-f, --csv_files`: List of CSV files to plot (required).
- `-g, --gpus`: Filter GPUs. List of number of GPUs to plot.
- `-d, --data_size`: Filter data sizes. List of data sizes to plot.
- `-fd, --filter_pdims`: List of pdims to filter (e.g., 1x4 2x2 4x8).
- `-ps, --pdims_strategy`: Strategy for plotting pdims (`plot_all` or `plot_fastest`).
  - `plot_all`: Plot every decomposition. 1xX and Xx1 as slabs, XxX as pencils.
  - `plot_fastest`: Plot the fastest decomposition.
- `-p, --precision`: Precision to filter by (`float32` or `float64`).
- `-fn, --function_name`: Function name to filter.
- `-ta, --time_aggregation`: Time aggregation method (`mean`, `min`, `max`).
- `-tc, --time_column`: Time column to plot (`jit_time`, `min_time`, `max_time`, `mean_time`, `std_div`, `last_time`).
- `-fs, --figure_size`: Figure size.
- `-nl, --nodes_in_label`: Use node names in labels.
- `-o, --output`: Output file (if none then only show plot).
- `-db, --dark_bg`: Use dark background for plotting.
- `-pd, --print_decompositions`: Print decompositions on plot (only for `plot_fastest`).
- `-b, --backends`: List of backends to include.
- `-sc, --scaling`: Scaling type (`Weak` or `Strong`).
