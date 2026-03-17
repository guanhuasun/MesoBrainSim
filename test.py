"""
test.py — GPU simulation test (100 ms) with 3D visualization.

Runs a WilsonCowan simulation on 1/4 of the isocortex population using the
CuPy (GPU) backend, then produces:
  - brain_activity.png  : static point cloud at final time step
  - brain_sim.gif       : animated signal propagation over time
"""

import time
import numpy as np
from mesobrainsim.simulation import Simulation

DATA_PATH   = "Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5"
N_FRACTION  = 0.25      # 1/4 of the population (~74k nodes)
MODEL       = "WilsonCowan"
SOLVER      = "Heun"
DT          = 0.1       # ms
T           = 100.0     # ms
RECORD_EVERY = 20       # store every 20 steps -> 50 frames

# ── 1. Run simulation ────────────────────────────────────────────────────────

print("=" * 60)
print("  MesoBrainSim GPU Test")
print("=" * 60)

sim = Simulation(
    data_path=DATA_PATH,
    subsample_fraction=N_FRACTION,
    model=MODEL,
    solver=SOLVER,
    dt=DT,
    T=T,
    record_every=RECORD_EVERY,
    use_gpu=True,
)

t0 = time.time()
result = sim.run()
elapsed = time.time() - t0

# Move trajectory to CPU numpy for reporting and visualization
def to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)

traj  = to_numpy(result.trajectory)   # (frames, N, state_dim)
times = to_numpy(result.times)

print()
print(f"  Nodes          : {len(result.anatomy)}")
print(f"  Edges          : {int(result.connectivity.W.nnz)}")
print(f"  Frames stored  : {traj.shape[0]}  (every {RECORD_EVERY} steps)")
print(f"  Trajectory     : {traj.shape}")
print(f"  Activity range : [{traj[:, :, 0].min():.4f}, {traj[:, :, 0].max():.4f}]")
print(f"  Elapsed        : {elapsed:.2f}s")
print()

# ── 2. Static visualization ──────────────────────────────────────────────────

print("Rendering static plot -> brain_activity.png ...")
import pyvista as pv
from mesobrainsim.viz import BrainVisualizer

viz = BrainVisualizer(result.anatomy.coords, traj, times)

pl = viz.plot_static(t_index=-1, cmap="hot", point_size=3.0, show=False)
pl.screenshot("brain_activity.png")
pl.close()
print("  Saved brain_activity.png")

# ── 3. Animated visualization ────────────────────────────────────────────────

print("Rendering animation -> brain_sim.gif ...")
viz.animate(output_path="brain_sim.gif", cmap="hot", point_size=3.0,
            framerate=15, step=1)
print("  Saved brain_sim.gif")

print()
print("Done.")
