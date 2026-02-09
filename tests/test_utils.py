import pandas as pd
import pytest

from jax_hpc_profiler.utils import (
    _parse_volume_query,
    _query_volume_type,
    parse_data_size_grep,
    probe_csv_metadata,
)


@pytest.fixture
def sample_df():
    """DataFrame with global_vol and local_vol columns for testing parse_data_size_grep."""
    data = {
        'x': [128, 128, 256, 100, 128],
        'y': [128, 128, 256, 200, 256],
        'z': [128, 128, 256, 300, 512],
        'px': [1, 2, 1, 1, 1],
        'py': [1, 2, 4, 1, 1],
        'gpus': [1, 4, 4, 1, 1],
    }
    df = pd.DataFrame(data)
    df['global_vol'] = df['x'] * df['y'] * df['z']
    df['local_vol'] = df['global_vol'] // df['gpus']
    return df


def test_parse_volume_query_global_int():
    col, vol = _parse_volume_query('global_2097152')
    assert col == 'global_vol'
    assert vol == 2097152


def test_parse_volume_query_global_nxmxk():
    col, vol = _parse_volume_query('global_128x128x128')
    assert col == 'global_vol'
    assert vol == 128 * 128 * 128


def test_parse_volume_query_local_int():
    col, vol = _parse_volume_query('local_524288')
    assert col == 'local_vol'
    assert vol == 524288


def test_parse_volume_query_local_nxmxk():
    col, vol = _parse_volume_query('local_128x64x64')
    assert col == 'local_vol'
    assert vol == 128 * 64 * 64


def test_parse_volume_query_invalid():
    with pytest.raises(ValueError, match='must start with global_ or local_'):
        _parse_volume_query('foobar')


def test_query_volume_type():
    assert _query_volume_type('global_128x128x128') == 'global_vol'
    assert _query_volume_type('local_256') == 'local_vol'


def test_parse_data_size_grep_global(sample_df):
    # global_vol for rows 0,1 = 128^3 = 2097152
    mask = parse_data_size_grep(sample_df, 'global_2097152')
    matches = sample_df[mask]
    assert len(matches) == 2
    assert all(matches['global_vol'] == 2097152)


def test_parse_data_size_grep_global_nxmxk(sample_df):
    # 128x128x128 = 2097152
    mask = parse_data_size_grep(sample_df, 'global_128x128x128')
    matches = sample_df[mask]
    assert len(matches) == 2
    assert all(matches['global_vol'] == 128**3)


def test_parse_data_size_grep_local(sample_df):
    # Row 0: local_vol = 128^3/1 = 2097152
    mask = parse_data_size_grep(sample_df, 'local_2097152')
    matches = sample_df[mask]
    assert len(matches) == 1
    assert matches.iloc[0]['local_vol'] == 2097152


def test_parse_data_size_grep_local_nxmxk(sample_df):
    # Row 0: local_vol = 128^3 = 2097152
    mask = parse_data_size_grep(sample_df, 'local_128x128x128')
    matches = sample_df[mask]
    assert len(matches) == 1
    assert matches.iloc[0]['local_vol'] == 128**3


def test_parse_data_size_grep_invalid(sample_df):
    with pytest.raises(ValueError, match='must start with global_ or local_'):
        parse_data_size_grep(sample_df, 'foobar')


def test_parse_data_size_grep_no_match(sample_df):
    mask = parse_data_size_grep(sample_df, 'global_999')
    assert not mask.any()


def test_probe_csv_metadata(tmp_path):
    csv_file = tmp_path / 'test_data.csv'
    data = [
        'fun1,float32,100,100,100,1,1,NCCL,1,0.1,1.0,1.2,1.1,0.01,1.1,1000,1000,1000,1000',
        'fun1,float32,100,100,100,2,1,NCCL,2,0.1,0.5,0.6,0.55,0.005,0.55,1000,1000,1000,1000',
    ]
    with open(csv_file, 'w') as f:
        f.write('\n'.join(data) + '\n')

    report = probe_csv_metadata([str(csv_file)])
    assert '=== CSV Probe Report ===' in report
    assert 'fun1' in report
    assert 'NCCL' in report
    assert 'float32' in report
    assert 'global_1000000' in report
