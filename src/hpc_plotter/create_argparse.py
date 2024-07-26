import argparse


def create_parser():
    """
    Create argument parser for the HPC Plotter package.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description='HPC Plotter for benchmarking data')
    parser.add_argument('-f',
                        '--csv_files',
                        nargs='+',
                        help='List of csv files to plot')
    parser.add_argument('-g',
                        '--gpus',
                        nargs='*',
                        type=int,
                        help='List of number of gpus to plot')
    parser.add_argument('-d',
                        '--data_size',
                        nargs='*',
                        type=int,
                        help='List of data size to plot')
    parser.add_argument('-fd',
                        '--pdims',
                        nargs='*',
                        help='List of pdims to filter, e.g., 1x4 2x2 4x8')
    parser.add_argument('-sc',
                        '--scaling',
                        choices=['Weak', 'Strong'],
                        required=True,
                        help='Scaling type (Weak or Strong)')
    parser.add_argument('-fs',
                        '--figure_size',
                        nargs=2,
                        type=int,
                        help='Figure size')
    parser.add_argument('-nl',
                        '--nodes_in_label',
                        action='store_true',
                        help='Use node names in labels')
    parser.add_argument('-o',
                        '--output',
                        help='Output file (if none then only show plot)',
                        default=None)
    parser.add_argument('-ta',
                        '--time_aggregation',
                        choices=['mean', 'min', 'max'],
                        default='mean',
                        help='Time aggregation method')
    parser.add_argument('-tc',
                        '--time_column',
                        choices=[
                            'jit_time', 'min_time', 'max_time', 'mean_time',
                            'std_div', 'last_time'
                        ],
                        default='mean_time',
                        help='Time column to plot')
    parser.add_argument('-db',
                        '--dark_bg',
                        type=bool,
                        default=False,
                        help='Use dark background for plotting')
    parser.add_argument('-pd',
                        '--print_decompositions',
                        action='store_true',
                        help='Print decompositions on plot')
    parser.add_argument('-b',
                        '--backends',
                        nargs='*',
                        default=['MPI', 'NCCL', 'MPI4JAX'],
                        help='List of backends to include')
    parser.add_argument('-p',
                        '--precision',
                        choices=['float32', 'float64'],
                        default='float32',
                        help='Precision to filter by (float32 or float64)')
    parser.add_argument('-t',
                        '--fft_type',
                        choices=['FFT', 'IFFT'],
                        default='FFT',
                        help='FFT type to filter by (FFT or IFFT)')
    parser.add_argument('-ps',
                        '--pdims_strategy',
                        choices=['plot_all', 'plot_fastest'],
                        default='plot_fastest',
                        help='Strategy for plotting pdims')
    parser.add_argument('--concat',
                        action='store_true',
                        help='Concatenate CSV files')
    parser.add_argument('--input',
                        type=str,
                        help='Input directory for concatenation')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Output directory for concatenation')

    return parser
