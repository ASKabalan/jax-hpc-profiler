import random

import numpy as np
import pandas as pd

# Constants
functions = ["FFT", "IFFT"]
precisions = ["float32", "float64"]
sizes = [1024, 2048, 4096, 8192, 16384]
px_py_combinations = [
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (8, 1),
    (1, 8),
    (4, 4),
    (8, 2),
    (2, 8),
    (1, 16),
    (16, 1),
    (8, 4),
    (4, 8),
    (1, 32),
    (32, 1),
    (8, 16),
    (16, 8),
    (1, 64),
    (64, 1),
    (16, 16),
    (1, 128),
    (128, 1),
    (1, 256),
    (256, 1),
    (128, 2),
    (2, 128),
]
backends = ["NCCL"]


# Calculate number of GPUs and nodes
def calculate_gpus_and_nodes(px, py):
    gpus = px * py
    nodes = gpus // 4
    return gpus, nodes


# Function to create the sample data with decompositions
def create_sample_data(filename):
    rows = []
    for function in functions:
        for precision in precisions:
            for size in sizes:
                for px, py in px_py_combinations:
                    gpus, nodes = calculate_gpus_and_nodes(px, py)
                    row = {
                        "function": function,
                        "precision": precision,
                        "x": size,
                        "y": size,
                        "z": size,
                        "px": px,
                        "py": py,
                        "backend": random.choice(backends),
                        "nodes": nodes,
                        "jit_time": round(random.uniform(10, 30), 2),
                        "min_time": round(random.uniform(5, 15), 2),
                        "max_time": round(random.uniform(15, 40), 2),
                        "mean_time": round(random.uniform(10, 30), 2),
                        "std_div": round(random.uniform(1, 5), 2),
                        "last_time": round(random.uniform(11, 25), 2),
                        "generated_code": random.randint(1000, 10000),
                        "argument_size": size,
                        "output_size": size // 2,
                        "temp_size": size // 4,
                        "flops": random.uniform(1e9, 2e9)
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)


# Creating three files
filenames = ["sample_data1.csv", "sample_data2.csv", "sample_data3.csv"]

for filename in filenames:
    create_sample_data(filename)
