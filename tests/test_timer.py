import jax
import jax.numpy as jnp
import numpy as np

from jax_hpc_profiler import AbstractTimer, JaxTimer, NoTimer, NumpyTimer, Timer


# Simple JAX function
@jax.jit
def simple_add(x, y):
    return x + y


# Simple numpy function
def numpy_add(x, y):
    return x + y


# --- Factory tests ---


def test_factory_default_returns_jax_timer():
    timer = Timer()
    assert isinstance(timer, JaxTimer)


def test_factory_jax_fn_false_returns_numpy_timer():
    timer = Timer(jax_fn=False)
    assert isinstance(timer, NumpyTimer)


def test_factory_active_false_returns_no_timer():
    timer = Timer(active=False)
    assert isinstance(timer, NoTimer)


def test_factory_active_false_ignores_jax_fn():
    timer = Timer(active=False, jax_fn=True)
    assert isinstance(timer, NoTimer)


# --- JaxTimer tests (existing tests adapted) ---


def test_timer_initialization():
    timer = Timer()
    assert isinstance(timer, JaxTimer)
    assert timer.save_jaxpr is False
    assert timer.compile_info is True
    assert timer.jit_time == 0.0
    assert len(timer.times) == 0


def test_chrono_jit():
    timer = Timer(save_jaxpr=True, compile_info=True)
    x = jnp.ones((10, 10))
    y = jnp.ones((10, 10))

    out = timer.chrono_jit(simple_add, x, y)
    assert out.shape == (10, 10)
    assert timer.jit_time > 0
    assert timer.compiled_code['JAXPR'] != 'N/A'
    assert timer.compiled_code['COMPILED'] != 'N/A'
    assert timer.profiling_data['generated_code'] != 'N/A'


def test_chrono_fun():
    timer = Timer()
    x = jnp.ones((10, 10))
    y = jnp.ones((10, 10))

    out = timer.chrono_fun(simple_add, x, y)
    assert out.shape == (10, 10)
    assert len(timer.times) == 1


def test_report(tmp_path):
    timer = Timer(save_jaxpr=False)
    x = jnp.ones((10, 10))
    y = jnp.ones((10, 10))

    timer.chrono_jit(simple_add, x, y)
    for _ in range(5):
        timer.chrono_fun(simple_add, x, y)

    csv_file = tmp_path / 'report.csv'
    md_file = tmp_path / 'report.md'

    # We specify nodes=1 explicitely to match what we expect in some tests, though default is 1
    timer.report(
        str(csv_file),
        function='simple_add',
        x=10,
        y=10,
        precision='float32',
        md_filename=str(md_file),
        extra_info={'custom_key': 'custom_val'},
    )

    assert csv_file.exists()
    assert md_file.exists()

    with open(csv_file) as f:
        content = f.read()
        assert 'simple_add' in content
        assert 'float32' in content

    with open(md_file) as f:
        content = f.read()
        assert '# Reporting for simple_add' in content
        assert 'custom_key' in content
        assert 'custom_val' in content


def test_normalize_memory_units():
    assert AbstractTimer._normalize_memory_units(100) == '100.00 B'
    assert AbstractTimer._normalize_memory_units(1024) == '1.00 KB'
    assert AbstractTimer._normalize_memory_units(1024**2) == '1.00 MB'
    assert AbstractTimer._normalize_memory_units(1024**3) == '1.00 GB'
    assert AbstractTimer._normalize_memory_units('N/A') == 'N/A'


def test_get_mean_times():
    timer = Timer()
    timer.times = [10.0, 20.0, 30.0]

    means = timer._get_mean_times()
    assert isinstance(means, np.ndarray)
    assert len(means) == 3
    assert means[0] == 10.0


# --- NumpyTimer tests ---


def test_numpy_timer_chrono_jit():
    timer = Timer(jax_fn=False)
    x = np.ones((10, 10))
    y = np.ones((10, 10))

    out = timer.chrono_jit(numpy_add, x, y)
    assert out.shape == (10, 10)
    assert timer.jit_time > 0


def test_numpy_timer_chrono_fun():
    timer = Timer(jax_fn=False)
    x = np.ones((10, 10))
    y = np.ones((10, 10))

    out = timer.chrono_fun(numpy_add, x, y)
    assert out.shape == (10, 10)
    assert len(timer.times) == 1
    assert timer.times[0] > 0


def test_numpy_timer_get_mean_times():
    timer = NumpyTimer()
    timer.times = [10.0, 20.0, 30.0]

    means = timer._get_mean_times()
    assert isinstance(means, np.ndarray)
    assert len(means) == 3
    np.testing.assert_array_equal(means, [10.0, 20.0, 30.0])


def test_numpy_timer_report(tmp_path):
    timer = Timer(jax_fn=False)
    x = np.ones((10, 10))
    y = np.ones((10, 10))

    timer.chrono_jit(numpy_add, x, y)
    for _ in range(3):
        timer.chrono_fun(numpy_add, x, y)

    csv_file = tmp_path / 'np_report.csv'
    md_file = tmp_path / 'np_report.md'

    timer.report(
        str(csv_file),
        function='numpy_add',
        x=10,
        precision='float64',
        md_filename=str(md_file),
    )

    assert csv_file.exists()
    assert md_file.exists()

    with open(csv_file) as f:
        content = f.read()
        assert 'numpy_add' in content
        assert 'float64' in content


# --- NoTimer tests ---


def test_no_timer_chrono_jit():
    timer = Timer(active=False)
    x = np.ones((10, 10))
    y = np.ones((10, 10))

    out = timer.chrono_jit(numpy_add, x, y)
    assert out.shape == (10, 10)
    assert timer.jit_time == 0.0
    assert len(timer.times) == 0


def test_no_timer_chrono_fun():
    timer = Timer(active=False)
    x = np.ones((10, 10))
    y = np.ones((10, 10))

    out = timer.chrono_fun(numpy_add, x, y)
    assert out.shape == (10, 10)
    assert len(timer.times) == 0


def test_no_timer_report_is_noop(tmp_path):
    timer = Timer(active=False)
    csv_file = tmp_path / 'noop.csv'

    timer.report(str(csv_file), function='test', x=10)

    assert not csv_file.exists()


# --- Reset tests ---


def test_reset_restores_initial_state():
    timer = Timer(jax_fn=False)
    x = np.ones((5,))
    y = np.ones((5,))

    timer.chrono_jit(numpy_add, x, y)
    timer.chrono_fun(numpy_add, x, y)

    assert timer.jit_time > 0
    assert len(timer.times) == 1

    timer._reset()

    assert timer.jit_time == 0.0
    assert timer.times == []
    assert timer.profiling_data == {
        'generated_code': 'N/A',
        'argument_size': 'N/A',
        'output_size': 'N/A',
        'temp_size': 'N/A',
    }
    assert timer.compiled_code == {'JAXPR': 'N/A', 'LOWERED': 'N/A', 'COMPILED': 'N/A'}


# --- Isinstance checks ---


def test_all_timers_are_abstract_timer():
    assert isinstance(JaxTimer(), AbstractTimer)
    assert isinstance(NumpyTimer(), AbstractTimer)
    assert isinstance(NoTimer(), AbstractTimer)
