from unittest.mock import MagicMock, patch

import pytest

from jax_hpc_profiler.plotting import (
    configure_axes,
    plot_by_data_size,
    plot_by_gpus,
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
    # function,precision,x,y,z,px,py,backend,nodes,jit,min,max,mean,std,last,gen_code,arg_size,
    # out_size,tmp_size
    data = [
        # Strong scaling rows: same (x,y,z)=100^3, different gpus
        'fun1,float32,100,100,100,1,1,NCCL,1,0.1,10.0,12.0,11.0,0.1,11.0,1000,1000,1000,1000',
        'fun1,float32,100,100,100,2,1,NCCL,2,0.1,5.0,6.0,5.5,0.05,5.5,1000,1000,1000,1000',
        'fun1,float32,100,100,100,2,2,NCCL,4,0.1,2.5,3.0,2.75,0.025,2.75,1000,1000,1000,1000',
        # Weak scaling rows: different (x,y,z) but same local_vol = 100^3
        # gpus=1, x=y=z=100 -> local_vol = 1000000
        # gpus=2, 200*100*100 = 2000000, local_vol = 1000000
        'fun1,float32,200,100,100,2,1,NCCL,2,0.1,10.5,12.5,11.5,0.15,11.5,2000,2000,2000,2000',
        # gpus=4, 400*100*100 = 4000000, local_vol = 1000000
        'fun1,float32,400,100,100,2,2,NCCL,4,0.1,11.0,13.0,12.0,0.2,12.0,4000,4000,4000,4000',
    ]
    with open(csv_file, 'w') as f:
        f.write('\n'.join(data) + '\n')
    return str(csv_file)


def test_plot_by_data_size_global(mock_plt, sample_csv):
    # Subplots per global volume (strong scaling), x=GPUs
    # Rows 0,1,2 have x=y=z=100, global_vol=1000000, gpus=1,2,4

    plot_by_data_size(
        csv_files=[sample_csv],
        data_size_queries=['global_1000000'],
        gpus=[1, 2, 4],
        functions=['fun1'],
        xlabel='GPUs',
        title='Strong Scaling',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_plot_by_data_size_local(mock_plt, sample_csv):
    # Subplots per local volume (weak scaling), x=GPUs
    # Rows 0 (gpus=1, 100^3), 3 (gpus=2, 200*100*100), 4 (gpus=4, 400*100*100)
    # all have local_vol = 1000000

    plot_by_data_size(
        csv_files=[sample_csv],
        data_size_queries=['local_1000000'],
        functions=['fun1'],
        xlabel='GPUs',
        title='Weak Scaling',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_plot_by_data_size_nxmxk(mock_plt, sample_csv):
    # Test NxMxK query format: 100x100x100 = 1000000
    plot_by_data_size(
        csv_files=[sample_csv],
        data_size_queries=['global_100x100x100'],
        gpus=[1, 2, 4],
        functions=['fun1'],
        xlabel='GPUs',
        title='Strong Scaling NxMxK',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_plot_by_gpus(mock_plt, sample_csv):
    # Subplots per GPU count, x=data size
    plot_by_gpus(
        csv_files=[sample_csv],
        gpus=[1, 2],
        data_size_queries=['global_1000000', 'global_2000000'],
        functions=['fun1'],
        xlabel='Data size',
        title='GPU counts',
    )

    assert mock_plt.show.called or mock_plt.savefig.called


def test_configure_axes():
    # Test the helper directly
    mock_ax = MagicMock()
    configure_axes(
        mock_ax, x_values=[1, 2, 4], y_values=[10, 5, 2.5], title='Test Plot', xlabel='X Label'
    )

    mock_ax.set_title.assert_called_with('Test Plot')
    mock_ax.set_xlabel.assert_called_with('X Label')
    mock_ax.set_yscale.assert_called_with('symlog')
