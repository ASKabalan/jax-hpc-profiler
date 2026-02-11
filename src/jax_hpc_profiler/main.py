from .create_argparse import create_argparser
from .plotting import plot_by_data_size, plot_by_gpus
from .utils import concatenate_csvs, probe_csv_metadata


def main():
    args = create_argparser()

    if args.command == 'concat':
        input_dir, output_dir = args.input, args.output
        concatenate_csvs(input_dir, output_dir)
    elif args.command == 'probe':
        report = probe_csv_metadata(args.csv_files)
        print(report)
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

        # Translate bare integers to global_NxNxN (cubed)
        data_size_queries = args.data_size
        if data_size_queries:
            translated = []
            for q in data_size_queries:
                if q.isdigit():
                    q = f'global_{q}x{q}x{q}'
                translated.append(q)
            data_size_queries = translated

        use_cube_notation = not args.disable_cube_notation

        if scaling in ('data', 'd'):
            plot_by_data_size(
                args.csv_files,
                args.gpus,
                data_size_queries,
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
                args.output,
                args.ideal_line,
                args.xscale,
                use_cube_notation=use_cube_notation,
            )
        elif scaling in ('gpus', 'g'):
            plot_by_gpus(
                args.csv_files,
                args.gpus,
                data_size_queries,
                args.function_name,
                args.precision,
                args.filter_pdims,
                args.pdim_strategy,
                args.print_decompositions,
                args.backends,
                args.plot_columns,
                args.memory_units,
                args.label_text,
                args.xlabel if getattr(args, 'xlabel', None) is not None else 'Data size',
                args.title if getattr(args, 'title', None) is not None else 'GPU counts',
                args.figure_size,
                args.output,
                args.ideal_line,
                args.xscale,
                use_cube_notation=use_cube_notation,
            )


if __name__ == '__main__':
    main()
