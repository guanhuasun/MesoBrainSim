# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An extensible, high-performance Python framework for simulating whole-brain dynamics in mice. The system integrates anatomical data, electrophysiological modeling, and structural connectivity, supporting both CPU (NumPy) and GPU (CuPy) backends.

## 1. Core Modules

### A. Anatomy Module
- **Purpose:** Defines the physical landscape of the simulation.
- **Data:** Load spatial coordinates (X, Y, Z) and metadata (Region ID, Brain Area name) for each node.
- **Implementation:** Subsample input data files to allow for scalable resolution (from coarse regional models to high-density point clouds).

### B. Electrophysiology Module
- **Purpose:** Defines the local dynamics of each node.
- **Model:** Neural Field / Neural Mass models (e.g., Wilson-Cowan or Jansen-Rit).
- **Features:** Must support region-specific and neuron-type-specific equations.

### C. Connectivity Module
- **Purpose:** Defines the communication network between nodes.
- **Data:** Load an adjacency matrix (Weights/Distances) representing structural connectivity.
- **Processing:** Subsample matrix dimensions to match the Anatomy module.

## 2. Technical Architecture

### Dual-Backend Engine
The framework abstracts the compute backend to allow seamless switching between CPU and GPU.
- **CPU Backend:** NumPy-based array operations.
- **GPU Backend:** CuPy-based array operations for CUDA acceleration.
- **Switching Mechanism:** A single global flag or configuration key (e.g., `USE_GPU = True/False`) that dynamically assigns the array provider (`xp`).

### Numerical Integration
Implement a solver (e.g., Euler or Heun's method) optimized for vectorized operations to handle thousands of nodes simultaneously.

## 3. Visualization
- **Package:** `PyVista`
- **Output:** 3D mesh or point cloud representation of the mouse brain post-simulation.
- **Dynamics:** Map simulation activity (e.g., voltage/firing rates) onto 3D nodes to visualize signal propagation over time.

## Data

- `Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5` (384 MB) — HDF5 file containing spatial coordinates, region metadata, and structural connectivity for the isocortex. Filename encodes parameters: `rho299` (density), `inh` (inhibitory neurons), `local` (local connectivity).
- The current data has **no weight information** in the connectivity matrix — all weights are initialized to 1.

## Environment

- **Python interpreter:** `C:\Users\sungu\anaconda3\python.exe`

## Implementation Plan

### Phase 1 — Project Scaffold & Backend Engine
- Create package structure: `mesobrainsim/` with `__init__.py`, `config.py`
- `config.py`: global `USE_GPU` flag; exports `xp` (either `numpy` or `cupy`) so all other modules import `xp` from here rather than importing numpy/cupy directly.

### Phase 2 — Anatomy Module (`mesobrainsim/anatomy.py`)
- Class `Anatomy` that loads the HDF5 file via `h5py`
- Reads X, Y, Z coordinates + Region ID + Brain Area name per node
- Subsampling: accept an `n_nodes` or `subsample_fraction` argument to downsample the full point cloud
- Output: structured array or dataframe of node metadata + coordinate array `(N, 3)`

### Phase 3 — Connectivity Module (`mesobrainsim/connectivity.py`)
- Class `Connectivity` that loads the adjacency matrix from the HDF5 file
- **No weight data in current file — initialize all weights to 1**
- Subsamples rows/columns to match the node indices selected by `Anatomy`
- Returns weight matrix `W (N, N)` and optional distance matrix `D (N, N)` as `xp` arrays

### Phase 4 — Electrophysiology Module (`mesobrainsim/ephys.py`)
- Abstract base class `NeuralModel` with a `dfdt(state, t)` method
- Concrete implementations:
  - `WilsonCowan` — excitatory/inhibitory population model
  - `JansenRit` — cortical column model
  - `IntegrateAndFire` — leaky integrate-and-fire spiking neuron
  - `HodgkinHuxley` — conductance-based spiking neuron (Na, K, leak channels)
  - `Kuramoto` — phase oscillator model for synchronization dynamics
- Support region-specific parameters: each node can carry its own parameter vector

### Phase 5 — Numerical Solver (`mesobrainsim/solver.py`)
- `EulerSolver` and `HeunSolver` classes (Heun = explicit trapezoidal, 2nd-order)
- Fully vectorized over N nodes using `xp` array ops
- Interface: `solver.run(model, W, T, dt)` → returns `state_trajectory (T_steps, N, state_dim)`

### Phase 6 — Visualization (`mesobrainsim/viz.py`)
- Class `BrainVisualizer` wrapping PyVista
- Takes node coordinates `(N, 3)` and a time-series activity array `(T, N)`
- Methods:
  - `plot_static(t)` — point cloud colored by activity at time step `t`
  - `animate(output_path)` — render an mp4/gif of signal propagation over time

### Phase 7 — Top-level Runner (`mesobrainsim/simulation.py`)
- `Simulation` class that wires all modules together
- Config-driven (dict or YAML): data path, subsample size, model type, solver, dt, T, USE_GPU
- Single `sim.run()` call → loads data → builds model → integrates → returns results

## Claude Code Strategy

### Performance & Architecture
- **Speed first.** Output must be most computationally efficient version. Favor vectorized ops (CuPy, NumPy, JAX) over loops.
- **GPU acceleration.** Use CuPy for all heavy numerical lifting. Minimize CPU-GPU transfers (`.get()`, `cp.array()`) — keep data on device.
- **Expert context.** User is peer-level expert. Do not explain algorithms; implement them. Use kernel fusions (`cp.ElementwiseKernel`) if it saves time.

### Coding Style
- **Max conciseness.** Shortest code that maintains peak performance. Use comprehensions, ternary operators, functional patterns.
- **No boilerplate.** Omit redundant setup, docstring fluff, verbose error handling unless critical to logic.

### Documentation & Git (Telegraphic Style)
- **Grammar sacrifice.** Omit articles, pronouns, formal punctuation in comments and commit messages.
- **Comment style:** `# init rand weights`, `# fix loop drift`, `# move to gpu`, `# fuse kernels`
- **Commit style:** `perf: opt cupy kernel`, `fix: reduce vram overhead`, `feat: add jax solver`

### Interaction Rules
- **No conversational filler.** Do not apologize or explain why code is fast. Just provide implementation.
- **Direct execution.** Prioritize shell commands that measure performance (`nvprof`, `time`, `pyinstrument`) immediately after refactoring.

## Suggested File Layout

```
mesobrainsim/
    __init__.py
    config.py          # xp backend switch
    anatomy.py
    connectivity.py
    ephys.py           # NeuralModel base + WilsonCowan, JansenRit, IntegrateAndFire, HodgkinHuxley, Kuramoto
    solver.py          # EulerSolver, HeunSolver
    viz.py             # PyVista visualization
    simulation.py      # top-level orchestrator
data/
    Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5
```
