import sys
from .utils import concatenate_csvs, clean_up_csv
from .plotting import plot_strong_scaling, plot_weak_scaling
from .create_argparse import create_parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.concat:
        if args.input and args.output_dir:
            concatenate_csvs(args.input, args.output_dir)
        else:
            print("Please provide input and output directories for concatenation.")
            sys.exit(1)
    else:
        dataframes = clean_up_csv(args.csv_files, args.precision, args.fft_type,
                                  args.gpus, args.data_size, args.pdims,
                                  args.pdims_strategy, args.time_aggregation,
                                  args.backends, args.time_column)

        if args.scaling == 'Weak':
            plot_weak_scaling(dataframes, args.gpus, args.nodes_in_label,
                              args.figure_size, args.output, args.dark_bg,
                              args.print_decompositions, args.backends,
                              args.pdims_strategy)
        elif args.scaling == 'Strong':
            plot_strong_scaling(dataframes, args.data_size, args.nodes_in_label,
                                args.figure_size, args.output, args.dark_bg,
                                args.print_decompositions, args.backends,
                                args.pdims_strategy)

if __name__ == "__main__":
    main()
