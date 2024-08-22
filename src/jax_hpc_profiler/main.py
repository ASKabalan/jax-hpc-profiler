import sys

from .create_argparse import create_argparser
from .plotting import plot_strong_scaling, plot_weak_scaling
from .utils import clean_up_csv, concatenate_csvs


def main():
    args = create_argparser()

    if args.command == 'concat':
        input_dir, output_dir = args.input, args.output
        concatenate_csvs(input_dir, output_dir)
    elif args.command == 'label_help':
        print(f"Customize the label text for the plot. using these commands.")
        print(' -- %m% or %methodname%: method name')
        print(' -- %f% or %function%: function name')
        print(' -- %pn% or %plot_name%: plot name')
        print(' -- %pr% or %precision%: precision')
        print(' -- %b% or %backend%: backend')
        print(' -- %p% or %pdims%: pdims')
        print(' -- %n% or %node%: node')
    elif args.command == 'plot':
        dataframes, available_gpu_counts, available_data_sizes = clean_up_csv(
            args.csv_files, args.precision, args.function_name, args.gpus,
            args.data_size, args.filter_pdims, args.pdim_strategy,
            args.backends, args.memory_units)
        if len(dataframes) == 0:
            print(f"No dataframes found for the given arguments. Exiting...")
            sys.exit(1)
        print(
            f"requested GPUS: {args.gpus} available GPUS: {available_gpu_counts}"
        )
        # filter back the requested data sizes and gpus
        
        args.gpus = available_gpu_counts if args.gpus is None else [gpu for gpu in args.gpus if gpu in available_gpu_counts]
        args.data_size = available_data_sizes if args.data_size is None else [data_size for data_size in args.data_size if data_size in available_data_sizes]

        if len(args.gpus) == 0:
            print(f"No dataframes found for the given GPUs. Exiting...")
            sys.exit(1)
        if len(args.data_size) == 0:
            print(f"No dataframes found for the given data sizes. Exiting...")
            sys.exit(1)

        if args.scaling == 'Weak':
            plot_weak_scaling(dataframes, args.gpus, args.figure_size,
                              args.output, args.dark_bg,
                              args.print_decompositions, args.backends,
                              args.precision, args.function_name,
                              args.plot_columns, args.memory_units,
                              args.label_text, args.pdim_strategy)
        elif args.scaling == 'Strong':
            plot_strong_scaling(dataframes, args.data_size, args.figure_size,
                                args.output, args.dark_bg,
                                args.print_decompositions, args.backends,
                                args.precision, args.function_name,
                                args.plot_columns, args.memory_units,
                                args.label_text, args.pdim_strategy)


if __name__ == "__main__":
    main()
