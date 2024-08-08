import jax

from jax_hpc_profiler import Timer


def fcn(m, n, k):
    return jax.numpy.dot(m, n) + k


timer = Timer(save_jaxpr=True)
m = jax.numpy.ones((1000, 1000))
n = jax.numpy.ones((1000, 1000))
k = jax.numpy.ones((1000, 1000))

timer.chrono_jit(fcn, m, n, k)
for i in range(10):
    timer.chrono_fun(fcn, m, n, k)

meta_data = {
    "function": "fcn",
    "precision": "float32",
    "x": 1000,
    "y": 1000,
    "z": 1000,
    "px": 1,
    "py": 1,
    "backend": "NCCL",
    "nodes": 1
}
extra_info = {"done": "yes"}

timer.report("examples/profiling/test.csv", **meta_data, extra_info=extra_info)
