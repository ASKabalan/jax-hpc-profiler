import sys

from .create_argparse import create_argparser
from .plotting import plot_strong_scaling, plot_weak_scaling
from .utils import clean_up_csv, concatenate_csvs


def main():
    parser = create_argparser()
    args = parser.parse_args()

    if args.command == 'concat':
        input_dir, output_dir = args.input, args.output
        concatenate_csvs(input_dir, output_dir)
    elif args.command == 'plot':
        dataframes = clean_up_csv(args.csv_files, args.precision,
                                  args.function_name, args.gpus,
                                  args.data_size, args.filter_pdims,
                                  args.pdims_strategy, args.time_aggregation,
                                  args.backends, args.time_column)

        if args.scaling == 'Weak':
            plot_weak_scaling(dataframes, args.gpus, args.nodes_in_label,
                              args.figure_size, args.output, args.dark_bg,
                              args.print_decompositions, args.backends,
                              args.pdims_strategy)
        elif args.scaling == 'Strong':
            plot_strong_scaling(dataframes, args.data_size,
                                args.nodes_in_label, args.figure_size,
                                args.output, args.dark_bg,
                                args.print_decompositions, args.backends,
                                args.pdims_strategy)


if __name__ == "__main__":
    main()
