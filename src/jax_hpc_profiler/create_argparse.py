import argparse


def create_argparser():
    """
    Create argument parser for the HPC Plotter package.

    Returns
    -------
    argparse.Namespace
        Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(
        description="HPC Plotter for benchmarking data")

    # Group for concatenation to ensure mutually exclusive behavior
    subparsers = parser.add_subparsers(dest="command", required=True)

    concat_parser = subparsers.add_parser("concat",
                                          help="Concatenate CSV files")
    concat_parser.add_argument("input",
                               type=str,
                               help="Input directory for concatenation")
    concat_parser.add_argument("output",
                               type=str,
                               help="Output directory for concatenation")

    # Arguments for plotting
    plot_parser = subparsers.add_parser("plot", help="Plot CSV data")
    plot_parser.add_argument("-f",
                             "--csv_files",
                             nargs="+",
                             help="List of CSV files to plot",
                             required=True)
    plot_parser.add_argument(
        "-g",
        "--gpus",
        nargs="*",
        type=int,
        help="List of number of GPUs to plot",
        default=None,
    )
    plot_parser.add_argument(
        "-d",
        "--data_size",
        nargs="*",
        type=int,
        help="List of data sizes to plot",
        default=None,
    )

    # pdims related arguments
    plot_parser.add_argument(
        "-fd",
        "--filter_pdims",
        nargs="*",
        help="List of pdims to filter, e.g., 1x4 2x2 4x8",
        default=None,
    )
    plot_parser.add_argument(
        "-ps",
        "--pdim_strategy",
        choices=["plot_all", "plot_fastest", "slab_yz", "slab_xy", "pencils"],
        nargs="*",
        default=["plot_fastest"],
        help="Strategy for plotting pdims",
    )

    # Function and precision related arguments
    plot_parser.add_argument(
        "-pr",
        "--precision",
        choices=["float32", "float64"],
        default=["float32", "float64"],
        nargs="*",
        help="Precision to filter by (float32 or float64)",
    )
    plot_parser.add_argument(
        "-fn",
        "--function_name",
        nargs="+",
        help="Function names to filter",
        default=None,
    )

    # Time or memory related arguments
    plotting_group = plot_parser.add_mutually_exclusive_group(required=True)
    plotting_group.add_argument(
        "-pt",
        "--plot_times",
        nargs="*",
        choices=[
            "jit_time",
            "min_time",
            "max_time",
            "mean_time",
            "std_time",
            "last_time",
        ],
        help="Time columns to plot",
    )
    plotting_group.add_argument(
        "-pm",
        "--plot_memory",
        nargs="*",
        choices=[
            "generated_code", "argument_size", "output_size", "temp_size"
        ],
        help="Memory columns to plot",
    )
    plot_parser.add_argument(
        "-mu",
        "--memory_units",
        default="GB",
        help="Memory units to plot (KB, MB, GB, TB)",
    )

    # Plot customization arguments
    plot_parser.add_argument("-fs",
                             "--figure_size",
                             nargs=2,
                             type=int,
                             help="Figure size",
                             default=(10, 6))
    plot_parser.add_argument("-o",
                             "--output",
                             help="Output file (if none then only show plot)",
                             default=None)
    plot_parser.add_argument("-db",
                             "--dark_bg",
                             action="store_true",
                             help="Use dark background for plotting")
    plot_parser.add_argument(
        "-pd",
        "--print_decompositions",
        action="store_true",
        help="Print decompositions on plot",
    )

    # Backend related arguments
    plot_parser.add_argument(
        "-b",
        "--backends",
        nargs="*",
        default=["MPI", "NCCL", "MPI4JAX"],
        help="List of backends to include",
    )

    # Scaling type argument
    plot_parser.add_argument(
        "-sc",
        "--scaling",
        choices=["Weak", "Strong", "w", "s"],
        required=True,
        help="Scaling type (Weak or Strong)",
    )

    # Label customization argument
    plot_parser.add_argument(
        "-l",
        "--label_text",
        type=str,
        help=
        ("Custom label for the plot. You can use placeholders: %%decomposition%% "
         "(or %%p%%), %%precision%% (or %%pr%%), %%plot_name%% (or %%pn%%), "
         "%%backend%% (or %%b%%), %%node%% (or %%n%%), %%methodname%% (or %%m%%)"
         ),
        default="%m%-%f%-%pn%-%pr%-%b%-%p%-%n%",
    )

    subparsers.add_parser("label_help", help="Label customization help")

    args = parser.parse_args()

    # if command was plot, then check if pdim_strategy is validat
    if args.command == "plot":
        if "plot_all" in args.pdim_strategy and len(args.pdim_strategy) > 1:
            print(
                "Warning: 'plot_all' strategy is combined with other strategies. Using 'plot_all' only."
            )
            args.pdim_strategy = ["plot_all"]

        if "plot_fastest" in args.pdim_strategy and len(
                args.pdim_strategy) > 1:
            print(
                "Warning: 'plot_fastest' strategy is combined with other strategies. Using 'plot_fastest' only."
            )
            args.pdim_strategy = ["plot_fastest"]
        if args.plot_times is not None:
            args.plot_columns = args.plot_times
        elif args.plot_memory is not None:
            args.plot_columns = args.plot_memory
        else:
            raise ValueError(
                "Either plot_times or plot_memory should be provided")

    return args
