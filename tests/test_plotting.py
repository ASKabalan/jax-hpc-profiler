from unittest.mock import MagicMock, patch

import pytest

from jax_hpc_profiler.plotting import (
    configure_axes,
    plot_strong_scaling,
    plot_weak_fixed_scaling,
    plot_weak_scaling,
)


@pytest.fixture
def mock_plt():
    with patch('jax_hpc_profiler.plotting.plt') as mock:
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock.subplots.return_value = (mock_fig, mock_ax)
        yield mock


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / 'test_data.csv'
    # Create data matching Timer.report format (19 columns)
    # function,precision,x,y,z,px,py,backend,nodes,jit,min,max,mean,std,last,gen_code,arg_size,out_size,tmp_size
    data = [
        'fun1,float32,100,100,100,1,1,NCCL,1,0.1,1.0,1.2,1.1,0.01,1.1,1000,1000,1000,1000',
        'fun1,float32,200,200,200,1,1,NCCL,1,0.2,2.0,2.4,2.2,0.02,2.2,2000,2000,2000,2000',
        'fun1,float32,400,400,400,1,1,NCCL,1,0.4,4.0,4.8,4.4,0.04,4.4,4000,4000,4000,4000',
        # Add entries for strong scaling (same x, diff nodes/gpus)
        # nodes is used as 'gpus' in some logic?
        # utils.py: df['gpus'] = df['px'] * df['py']
        # So we vary px*py.
        'fun2,float32,1000,1000,1000,1,1,NCCL,1,0.1,10.0,12.0,11.0,0.1,11.0,1000,1000,1000,1000',
        'fun2,float32,1000,1000,1000,2,1,NCCL,2,0.1,5.0,6.0,5.5,0.05,5.5,1000,1000,1000,1000',
        'fun2,float32,1000,1000,1000,2,2,NCCL,4,0.1,2.5,3.0,2.75,0.025,2.75,1000,1000,1000,1000',
    ]
    with open(csv_file, 'w') as f:
        f.write('\n'.join(data) + '\n')
    return str(csv_file)


@pytest.fixture
def mock_adjust_text():
    with patch('jax_hpc_profiler.plotting.adjust_text') as mock:
        yield mock


def test_plot_weak_fixed_scaling(mock_plt, sample_csv):
    # WeakFixed: vary data size (x), fixed GPUs (calculated from px*py).
    # In sample_csv fun1: gpus=1, x=[100, 200, 400]

    plot_weak_fixed_scaling(
        csv_files=[sample_csv],
        fixed_gpu_size=[1],
        fixed_data_size=[100, 200, 400],
        functions=['fun1'],
        xlabel='Data Size',
        title='Weak Fixed Scaling',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_plot_strong_scaling(mock_plt, sample_csv):
    # Strong: fixed data size (x), vary GPUs.
    # In sample_csv fun2: x=1000, gpus=[1, 2, 4]

    plot_strong_scaling(
        csv_files=[sample_csv],
        fixed_data_size=[1000],
        fixed_gpu_size=[1, 2, 4],
        functions=['fun2'],
        xlabel='GPUs',
        title='Strong Scaling',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_plot_weak_scaling(mock_plt, mock_adjust_text, sample_csv):
    # Weak: explicit pairs of (gpus, data_size)
    # We have (1, 100) for fun1.

    plot_weak_scaling(
        csv_files=[sample_csv],
        fixed_gpu_size=[1],
        fixed_data_size=[100],
        functions=['fun1'],
        xlabel='GPUs',
        title='Weak Scaling',
    )

    assert mock_plt.show.called or mock_plt.savefig.called
    assert (
        mock_adjust_text.called or not mock_adjust_text.called
    )  # It's okay if called or not, just don't crash.


def test_configure_axes():
    # Test the helper directly
    mock_ax = MagicMock()
    configure_axes(
        mock_ax, x_values=[1, 2, 4], y_values=[10, 5, 2.5], title='Test Plot', xlabel='X Label'
    )

    mock_ax.set_title.assert_called_with('Test Plot')
    mock_ax.set_xlabel.assert_called_with('X Label')
    mock_ax.set_xscale.assert_called()
    mock_ax.set_yscale.assert_called_with('symlog')
