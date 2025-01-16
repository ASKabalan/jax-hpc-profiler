#!/usr/bin/env python
# coding: utf-8

# In[1]:

import jax
import jax.numpy as jnp
import numpy as np

from jax_hpc_profiler import Timer

# In[ ]:


# Define JAX functions
@jax.jit
def mma(m, n, k):
    return jnp.dot(m, n) + k


@jax.jit
def inv(m):
    return jnp.linalg.inv(m)


# Define NumPy functions
def numpy_mma(m, n, k):
    return np.dot(m, n) + k


def numpy_inv(m):
    return np.linalg.inv(m)


# In[7]:

# Initialize timers
jax_timer = Timer(save_jaxpr=True, jax_fn=True)
numpy_timer = Timer(save_jaxpr=False, jax_fn=False)

# Profiling matrix sizes from 100x100 to 1000x1000
for size in range(100, 1001, 100):
    # JAX Matrices
    m_jax = jnp.ones((size, size))
    n_jax = jnp.ones((size, size))
    k_jax = jnp.ones((size, size))
    rand_jax = jax.random.normal(jax.random.key(0), (size, size))

    # NumPy Matrices
    m_np = np.ones((size, size))
    n_np = np.ones((size, size))
    k_np = np.ones((size, size))
    rand_np = np.random.rand(size, size)

    # --- JAX profiling ---
    jax_timer.chrono_jit(mma, m_jax, n_jax, k_jax)
    for _ in range(10):
        jax_timer.chrono_fun(mma, m_jax, n_jax, k_jax)

    kwargs = {"function": "mma", "precision": "float32", "x": size, "y": size}
    extra_info = {"done": "yes"}
    jax_timer.report("JAX.csv", **kwargs, extra_info=extra_info)

    jax_timer.chrono_jit(inv, rand_jax)
    for _ in range(10):
        jax_timer.chrono_fun(inv, rand_jax)

    kwargs = {"function": "inv", "precision": "float32", "x": size, "y": size}
    jax_timer.report("JAX.csv", **kwargs, extra_info=extra_info)

    # --- NumPy profiling ---
    numpy_timer.chrono_jit(numpy_mma, m_np, n_np, k_np)
    for _ in range(10):
        numpy_timer.chrono_fun(numpy_mma, m_np, n_np, k_np)

    kwargs = {"function": "mma", "precision": "float32", "x": size, "y": size}
    numpy_timer.report("NUMPY.csv", **kwargs, extra_info=extra_info)

    numpy_timer.chrono_jit(numpy_inv, rand_np)
    for _ in range(10):
        numpy_timer.chrono_fun(numpy_inv, rand_np)

    kwargs = {"function": "inv", "precision": "float32", "x": size, "y": size}
    numpy_timer.report("NUMPY.csv", **kwargs, extra_info=extra_info)

# In[ ]:

import matplotlib.pyplot as plt
import seaborn as sns

from jax_hpc_profiler.plotting import plot_strong_scaling, plot_weak_scaling

plt.rcParams.update({'font.size': 15})

sns.set_context("talk")

csv_file = ["NUMPY.csv", "JAX.csv"]

plot_weak_scaling(csv_files=csv_file,
                  figure_size=(12, 8),
                  label_text="%m%-%f%")

# In[20]:

import matplotlib.pyplot as plt
import seaborn as sns

from jax_hpc_profiler.plotting import plot_strong_scaling, plot_weak_scaling

plt.rcParams.update({'font.size': 15})

sns.set_context("talk")

csv_file = ["NUMPY.csv", "JAX.csv"]

plot_weak_scaling(csv_files=csv_file,
                  fixed_data_size=np.arange(100, 701, 100).tolist(),
                  figure_size=(12, 8),
                  label_text="%m%-%f%")

# In[25]:

get_ipython().system(
    'jhp plot -f NUMPY.csv JAX.csv -d 100 200 300 -sc w -pt mean_time -o weak_scaling.png -l "%m%-%f%"'
)

# In[26]:


@jax.jit
def multi_out(x, y):
    return x + y, x - y


timer = Timer()

timer.chrono_jit(multi_out, 1, 2, ndarray_arg=0)
