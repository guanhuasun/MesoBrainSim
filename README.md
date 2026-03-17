# MesoBrainSim

A high-performance Python framework for simulating whole-brain dynamics in mice. Integrates anatomical data, electrophysiological modeling, and structural connectivity with support for both CPU (NumPy) and GPU (CuPy) backends.

## Features

- **Anatomy:** Load and subsample spatial coordinates + region metadata from Allen Brain Cell Atlas HDF5 files
- **Connectivity:** Structural adjacency matrix with distance support
- **Electrophysiology models:** Wilson-Cowan, Jansen-Rit, Integrate-and-Fire, Hodgkin-Huxley, Kuramoto
- **Solvers:** Euler and Heun (2nd-order) vectorized over thousands of nodes
- **Visualization:** 3D point cloud and animated signal propagation via PyVista

## Installation

```bash
pip install numpy h5py pyvista
# For GPU support:
pip install cupy-cuda12x  # match your CUDA version
```

## Quick Start

```python
from mesobrainsim.simulation import Simulation

sim = Simulation(
    data_path="data/Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5",
    n_nodes=1000,
    model="WilsonCowan",
    solver="Heun",
    dt=0.1,
    T=500.0,
    use_gpu=False,
)
results = sim.run()
results.visualize()
```

## Project Structure

```
mesobrainsim/
    config.py       # backend switch (NumPy / CuPy)
    anatomy.py      # spatial coordinates and region metadata
    connectivity.py # structural adjacency matrix
    ephys.py        # neural models
    solver.py       # numerical integrators
    viz.py          # PyVista 3D visualization
    simulation.py   # top-level orchestrator
data/               # HDF5 input files
```
