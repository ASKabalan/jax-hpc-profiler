import argparse


def create_argparser():
    """
    Create argument parser for the HPC Plotter package.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description='HPC Plotter for benchmarking data')

    # Group for concatenation to ensure mutually exclusive behavior
    subparsers = parser.add_subparsers(dest='command', required=True)

    concat_parser = subparsers.add_parser('concat',
                                          help='Concatenate CSV files')
    concat_parser.add_argument('input',
                               type=str,
                               help='Input directory for concatenation')
    concat_parser.add_argument('output',
                               type=str,
                               help='Output directory for concatenation')

    # Arguments for plotting
    plot_parser = subparsers.add_parser('plot', help='Plot CSV data')
    plot_parser.add_argument('-f',
                             '--csv_files',
                             nargs='+',
                             help='List of CSV files to plot',
                             required=True)
    plot_parser.add_argument('-g',
                             '--gpus',
                             nargs='*',
                             type=int,
                             help='List of number of GPUs to plot')
    plot_parser.add_argument('-d',
                             '--data_size',
                             nargs='*',
                             type=int,
                             help='List of data sizes to plot')

    # pdims related arguments
    plot_parser.add_argument('-fd',
                             '--filter_pdims',
                             nargs='*',
                             help='List of pdims to filter, e.g., 1x4 2x2 4x8')
    plot_parser.add_argument('-ps',
                             '--pdims_strategy',
                             choices=['plot_all', 'plot_fastest'],
                             default='plot_fastest',
                             help='Strategy for plotting pdims')

    # Function and precision related arguments
    plot_parser.add_argument(
        '-p',
        '--precision',
        choices=['float32', 'float64'],
        default='float32',
        help='Precision to filter by (float32 or float64)')
    plot_parser.add_argument('-fn',
                             '--function_name',
                             default='FFT',
                             help='Function name to filter')

    # Time related arguments
    plot_parser.add_argument('-ta',
                             '--time_aggregation',
                             choices=['mean', 'min', 'max'],
                             default='mean',
                             help='Time aggregation method')
    plot_parser.add_argument('-tc',
                             '--time_column',
                             choices=[
                                 'jit_time', 'min_time', 'max_time',
                                 'mean_time', 'std_div', 'last_time'
                             ],
                             default='mean_time',
                             help='Time column to plot')

    # Plot customization arguments
    plot_parser.add_argument('-fs',
                             '--figure_size',
                             nargs=2,
                             type=int,
                             help='Figure size')
    plot_parser.add_argument('-nl',
                             '--nodes_in_label',
                             action='store_true',
                             help='Use node names in labels')
    plot_parser.add_argument('-o',
                             '--output',
                             help='Output file (if none then only show plot)',
                             default=None)
    plot_parser.add_argument('-db',
                             '--dark_bg',
                             type=bool,
                             default=False,
                             help='Use dark background for plotting')
    plot_parser.add_argument('-pd',
                             '--print_decompositions',
                             action='store_true',
                             help='Print decompositions on plot')

    # Backend related arguments
    plot_parser.add_argument('-b',
                             '--backends',
                             nargs='*',
                             default=['MPI', 'NCCL', 'MPI4JAX'],
                             help='List of backends to include')

    # Scaling type argument
    plot_parser.add_argument('-sc',
                             '--scaling',
                             choices=['Weak', 'Strong'],
                             required=True,
                             help='Scaling type (Weak or Strong)')

    return parser
