# Poly2Graph 
<!-- [📄 *arXiv*](https://arxiv.org/) -->

`Poly2Graph` is a Python package for automatic *non-Hermitian spectral graph* construction. It takes in the characteristic polynomial and returns the spectral graph.

> Topological physics is one of the most dynamic and rapidly advancing fields in modern physics. Conventionally, topological classification focuses on eigenstate windings, a concept central to Hermitian topological lattices (e.g., topological insulators). Beyond such notion of topology, we unravel a distinct and diverse graph topology emerging in non-Hermitian systems' energy spectra, featuring a kaleidoscope of exotic shapes like stars, kites, insects, and braids. The spectral graph solely depends on the algebraic form of characteristic polynomial.

<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/poly2graph_dev/main/assets/SGs_demo.png" width="800" />
</p>

## Features
- **Poly2Graph**
  1. Fast construction of spectral graph from and one-dimensional models
  2. Support for generic one-band and multi-band models
  3. Adaptive resolution to reduce floating operation cost and memory usage
  4. Automatic spectral boundary inference
  5. Convert a skeleton image to its graph representation
  <!-- 6. Dataset generation
  1. Visualization of spectral potential, density of states, and spectral graph -->

## Installation

First make sure you have installed [`tensorflow`](https://www.tensorflow.org/) according to your machine specifics. This module is tested on `Python >= 3.12`, `tensorflow >=2.10`.

`tensorflow` is required for the optimization of computation bottleneck.

You can install the package via pip:

```bash
$ pip install poly2graph
```

or clone the repository and install it manually:

```bash
$ git clone https://github.com/sarinstein-yan/poly2graph.git
$ cd poly2graph
$ pip install .
```

Check the installation:

```python
import poly2graph as p2g
print(p2g.__version__)
```

## Usage

See the [Poly2Graph Tutorial JupyterNotebook](https://github.com/sarinstein-yan/poly2graph_dev/blob/main/poly2graph_demo.ipynb).

```python
import numpy as np
import networkx as nx
import tensorflow as tf
import sympy as sp
from sympy.polys.polytools import Poly
import matplotlib.pyplot as plt

# always start by initializing the symbols for k, z, and E
k = sp.symbols('k', real=True)
z, E = sp.symbols('z E', complex=True)
```

### A generic **one-band** example:

characteristic polynomial:

$$P(E,z) := h(z) - E = z^4 -z -z^{-2} -E$$

Its Bloch Hamiltonian (Fourier transformed Hamiltonian in momentum space) is a scalar function:

$$h(z) = z^4 - z - z^{-2}$$

where the phase factor is defined as $z:=e^{ik}$.

Expressed in terms of crystal momentum $k$:

$$h(k) = e^{4ik} - e^{ik} - e^{-2ik}$$

---
The valid input formats to initialize a `p2g.SpectralGraph` object are:
1. Characteristic polynomial in terms of `z` and `E`:
   - as a string of the Poly in terms of `z` and `E`
   - as a `sympy`'s `Poly` (`sympy.polys.polytools.Poly`) with {`z`, `1/z`, `E`} as generators
2. Bloch Hamiltonian in terms of `k` or `z`
   - as a `sympy` `Matrix` in terms of `k`
   - as a `sympy` `Matrix` in terms of `z`

All the following `characteristic`s are valid and will initialize to the same characteristic polynomial and therefore produce the same spectral graph:
```python
char_poly_str = '-z**-2 - E - z + z**4'

char_poly_Poly = Poly(
    -z**-2 - E - z + z**4,
    z, 1/z, E # generators are z, 1/z, E
)

phase_k = sp.exp(sp.I*k)
char_hamil_k = sp.Matrix([-phase_k**2 - phase_k + phase_k**4])

char_hamil_z = sp.Matrix([-z**-2 - E - z + z**4])
```

Let us just use the string to initialize and see a set of properties that are computed automatically

```python
sg = p2g.SpectralGraph(char_poly_str, k=k, z=z, E=E)
```
<p align="center">
    <img src="https://raw.githubusercontent.com/sarinstein-yan/poly2graph_dev/main/assets/spectral_graph_one_band.png" width="300" />
</p>