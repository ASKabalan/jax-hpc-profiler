import sys
from typing import List, Optional

from .create_argparse import create_argparser
from .plotting import plot_strong_scaling, plot_weak_scaling
from .utils import clean_up_csv, concatenate_csvs


def main():
    args = create_argparser()

    if args.command == "concat":
        input_dir, output_dir = args.input, args.output
        concatenate_csvs(input_dir, output_dir)
    elif args.command == "label_help":
        print(f"Customize the label text for the plot. using these commands.")
        print(" -- %m% or %methodname%: method name")
        print(" -- %f% or %function%: function name")
        print(" -- %pn% or %plot_name%: plot name")
        print(" -- %pr% or %precision%: precision")
        print(" -- %b% or %backend%: backend")
        print(" -- %p% or %pdims%: pdims")
        print(" -- %n% or %node%: node")
    elif args.command == "plot":

        if args.scaling.lower() == "weak" or args.scaling.lower() == "w":
            plot_weak_scaling(
                args.csv_files,
                args.gpus,
                args.data_size,
                args.function_name,
                args.precision,
                args.filter_pdims,
                args.pdim_strategy,
                args.print_decompositions,
                args.backends,
                args.plot_columns,
                args.memory_units,
                args.label_text,
                args.figure_size,
                args.dark_bg,
                args.output,
            )
        elif args.scaling.lower() == "strong" or args.scaling.lower() == "s":
            plot_strong_scaling(
                args.csv_files,
                args.gpus,
                args.data_size,
                args.function_name,
                args.precision,
                args.filter_pdims,
                args.pdim_strategy,
                args.print_decompositions,
                args.backends,
                args.plot_columns,
                args.memory_units,
                args.label_text,
                args.figure_size,
                args.dark_bg,
                args.output,
            )


if __name__ == "__main__":
    main()
