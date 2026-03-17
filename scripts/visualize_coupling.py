"""
Render coupling-sweep GIFs for c in [0.0, 0.5, 2.0, 5.0].
Reads h5 files produced by test_coupling.py (no re-simulation).
Two GIFs per c:
  coupling_{tag}_full.gif   — clim=(-65, 40)  shows spike wavefront
  coupling_{tag}_sub.gif    — clim=(-66, -62)  reveals subthreshold propagation
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py

from mesobrainsim.anatomy import Anatomy
from mesobrainsim.viz import BrainVisualizer

DATA  = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
FRAC  = 0.05
SEED  = 0
C_VIZ = [0.0, 0.5, 2.0, 5.0]

print("Loading anatomy for coordinates...")
anatomy = Anatomy(DATA, subsample_fraction=FRAC, seed=SEED)
coords  = anatomy.coords   # (N, 3) in µm, z not yet flipped (viz does that)
N       = len(anatomy)
print(f"N={N}")

for c in C_VIZ:
    tag = f"c{c}".replace('.', 'p')
    h5  = f"coupling_sweep_{tag}.h5"
    if not os.path.exists(h5):
        print(f"[{tag}] {h5} not found — run test_coupling.py first")
        continue

    with h5py.File(h5, 'r') as f:
        data  = f['probes/V/data'][:]       # (T, N)
        times = f['metadata/times'][:]      # (T,)

    traj = data[:, :, np.newaxis].astype(np.float32)  # (T, N, 1)

    # ── full-range: spike wavefront visible ──────────────────────────────────
    out_full = f"coupling_{tag}_full.gif"
    print(f"[{tag}] Rendering {out_full}  (clim -65..40) ...")
    viz = BrainVisualizer(coords, traj, times)
    viz.animate(out_full, cmap='turbo', point_size=3.0,
                framerate=15, clim=(-65, 40))

    # ── subthreshold: propagation into silent region ──────────────────────────
    out_sub = f"coupling_{tag}_sub.gif"
    print(f"[{tag}] Rendering {out_sub}  (clim -66..-62) ...")
    viz2 = BrainVisualizer(coords, traj, times)
    viz2.animate(out_sub, cmap='turbo', point_size=3.0,
                 framerate=15, clim=(-66.0, -62.0))

print("\nDone. GIFs:")
for c in C_VIZ:
    tag = f"c{c}".replace('.', 'p')
    print(f"  coupling_{tag}_full.gif   coupling_{tag}_sub.gif")
