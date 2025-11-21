from .create_argparse import create_argparser
from .plotting import plot_strong_scaling, plot_weak_fixed_scaling, plot_weak_scaling
from .utils import concatenate_csvs


def main():
    args = create_argparser()

    if args.command == 'concat':
        input_dir, output_dir = args.input, args.output
        concatenate_csvs(input_dir, output_dir)
    elif args.command == 'label_help':
        print('Customize the label text for the plot. using these commands.')
        print(' -- %m% or %methodname%: method name')
        print(' -- %f% or %function%: function name')
        print(' -- %pn% or %plot_name%: plot name')
        print(' -- %pr% or %precision%: precision')
        print(' -- %b% or %backend%: backend')
        print(' -- %p% or %pdims%: pdims')
        print(' -- %n% or %node%: node')
    elif args.command == 'plot':
        scaling = args.scaling.lower()
        if scaling in ('weak', 'w'):
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
                args.xlabel if getattr(args, 'xlabel', None) is not None else 'Number of GPUs',
                args.title,
                args.figure_size,
                args.dark_bg,
                args.output,
                args.weak_ideal_line,
                args.weak_reverse_axes,
            )
        elif scaling in ('strong', 's'):
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
                args.xlabel if getattr(args, 'xlabel', None) is not None else 'Number of GPUs',
                args.title if getattr(args, 'title', None) is not None else 'Data sizes',
                args.figure_size,
                args.dark_bg,
                args.output,
            )
        elif scaling in ('weakfixed', 'wf'):
            plot_weak_fixed_scaling(
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
                args.xlabel if getattr(args, 'xlabel', None) is not None else 'Data sizes',
                args.title if getattr(args, 'title', None) is not None else 'Number of GPUs',
                args.figure_size,
                args.dark_bg,
                args.output,
            )


if __name__ == '__main__':
    main()
