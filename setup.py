from setuptools import setup, find_packages

setup(
    name='hpc_plotter',
    version='0.1.0',
    description='HPC Plotter for benchmarking data',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'jax'
    ],
    entry_points={
        'console_scripts': [
            'hpc-plotter=hpc_plotter.main:main',
        ],
    },
)
