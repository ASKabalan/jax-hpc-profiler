import jax
import jax.numpy as jnp
import numpy as np

from jax_hpc_profiler import Timer


# Simple JAX function
@jax.jit
def simple_add(x, y):
    return x + y


def test_timer_initialization():
    timer = Timer()
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

    # Run once to compile (if using jit externally, but here we just call the function)
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
    timer = Timer()
    # Mocking internal state as if we had data
    timer.jax_fn = True
    timer.compile_info = True

    assert timer._normalize_memory_units(100) == '100.00 B'
    assert timer._normalize_memory_units(1024) == '1.00 KB'
    assert timer._normalize_memory_units(1024**2) == '1.00 MB'
    assert timer._normalize_memory_units(1024**3) == '1.00 GB'


def test_get_mean_times():
    timer = Timer()
    timer.times = [10.0, 20.0, 30.0]

    means = timer._get_mean_times()
    assert isinstance(means, np.ndarray)
    assert len(means) == 3
    assert means[0] == 10.0
