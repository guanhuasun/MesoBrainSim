"""
Simulation: constant stimulation of VISp4 (Allen CCF region_id=593) for 500ms.
Saves: recordings_visp4.h5, visp4_activity.gif, visp4_timeseries.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mesobrainsim.anatomy import Anatomy
from mesobrainsim.connectivity import Connectivity
from mesobrainsim.stimulation import ConstantStimulator
from mesobrainsim.measurement import MeasurementHook, Probe
from mesobrainsim.simulation import Simulation
from mesobrainsim.utils import resolve_nodes

# ── config ──────────────────────────────────────────────────────────────────
DATA    = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
FRAC    = 0.25        # subsample fraction
SEED    = 0
DT      = 0.025       # ms (HH requires small dt)
T       = 500.0       # ms
REC     = 200         # record every 200 steps -> 101 frames
MAG     = 15.0        # µA/cm² — drives ~60Hz firing in stim nodes
STIM_ID = 593         # VISp4 Allen CCF region_id
TAG     = 'visp4'
MODEL_KW = dict(I_ext=0.0, c=0.5)  # gap junction; c=0.5 drives propagation
# ────────────────────────────────────────────────────────────────────────────

n_steps    = int(T / DT)
n_recorded = n_steps // REC + 1

print(f"[{TAG}] Pre-building anatomy (frac={FRAC}, seed={SEED})...")
anatomy      = Anatomy(DATA, subsample_fraction=FRAC, seed=SEED)
connectivity = Connectivity(DATA, anatomy)
N = len(anatomy)
print(f"[{TAG}] N={N} nodes")

stim_idx = resolve_nodes(anatomy, STIM_ID)
print(f"[{TAG}] VISp4 nodes in subsample: {len(stim_idx)}")
if len(stim_idx) == 0:
    print(f"[{TAG}] WARNING: no VISp4 nodes found — check region_id. "
          f"Sample region_ids: {np.unique(anatomy.region_ids)[:10]}")

# ── probes: whole population (E variable) ───────────────────────────────────
probe = Probe('whole_pop_E', np.arange(N), var_index=0)
mhook = MeasurementHook(
    [probe], f'recordings_{TAG}.h5',
    anatomy, connectivity.W, n_recorded
)

# ── stimulator ───────────────────────────────────────────────────────────────
stim = ConstantStimulator(
    anatomy, selector=STIM_ID,
    magnitude=MAG, var_col=0,
    t_start=0.0, t_end=T
)

# ── run ──────────────────────────────────────────────────────────────────────
sim = Simulation(
    data_path=DATA, subsample_fraction=FRAC, seed=SEED,
    model='HodgkinHuxley', model_kwargs=MODEL_KW, solver='Heun',
    dt=DT, T=T, record_every=REC,
    use_gpu=False,
    stimulator=stim,
    measurement_hooks=[mhook],
    return_trajectory=True,
)
result = sim.run()
traj  = result.trajectory   # (101, N, 2)
times = result.times        # (101,)

# ── 3D animated GIF ──────────────────────────────────────────────────────────
print(f"[{TAG}] Rendering 3D animation -> {TAG}_activity.gif ...")
from mesobrainsim.viz import BrainVisualizer
viz = BrainVisualizer(anatomy.coords, traj, times)
viz.animate(output_path=f'{TAG}_activity.gif', cmap='turbo', point_size=3.0,
            framerate=12, step=1, clim=(-65, 40))

# ── time-series plot from h5 ──────────────────────────────────────────────────
print(f"[{TAG}] Plotting time series...")
import h5py

with h5py.File(f'recordings_{TAG}.h5', 'r') as f:
    data  = f['probes/whole_pop_E/data'][:]   # (101, N)
    t_rec = f['metadata/times'][:]            # (101,)

# all-node mean E
mean_E_all = data.mean(axis=1)

# stimulated-region mean E
if len(stim_idx) > 0:
    mean_E_stim = data[:, stim_idx].mean(axis=1)
else:
    mean_E_stim = np.full(len(t_rec), np.nan)

# non-stimulated mean E
mask_non = np.ones(N, dtype=bool)
if len(stim_idx) > 0:
    mask_non[stim_idx] = False
mean_E_non = data[:, mask_non].mean(axis=1) if mask_non.any() else np.full(len(t_rec), np.nan)

fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# ── top: mean activity traces ──
ax = axes[0]
ax.plot(t_rec, mean_E_all,  lw=1.5, label='Whole population', color='steelblue')
ax.plot(t_rec, mean_E_stim, lw=2.0, label=f'VISp4 (id={STIM_ID}, n={len(stim_idx)})',
        color='crimson', linestyle='--')
ax.plot(t_rec, mean_E_non,  lw=1.0, label='Non-stimulated', color='gray', alpha=0.7)
ax.axvspan(0, T, alpha=0.05, color='crimson', label='Stim window')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Mean E activity')
ax.set_title('VISp4 Constant Stimulation — Whole Population E Activity')
ax.legend(fontsize=8)
ax.set_xlim(0, T)

# ── bottom: spatial activity snapshot at final frame ──
ax2 = axes[1]
final_E = data[-1]        # (N,)
scatter = ax2.scatter(
    anatomy.coords[:, 0], anatomy.coords[:, 1],
    c=final_E, cmap='turbo', s=0.5, vmin=-65, vmax=40,
)
if len(stim_idx) > 0:
    ax2.scatter(
        anatomy.coords[stim_idx, 0], anatomy.coords[stim_idx, 1],
        s=4, facecolors='none', edgecolors='cyan', linewidths=0.5,
        label=f'VISp4 nodes'
    )
    ax2.legend(fontsize=8, markerscale=3)
fig.colorbar(scatter, ax=ax2, label='E activity')
ax2.set_xlabel('X (µm)')
ax2.set_ylabel('Y (µm)')
ax2.set_title(f'E Activity at t={t_rec[-1]:.0f} ms (coronal projection)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{TAG}_timeseries.png', dpi=150, bbox_inches='tight')
print(f"[{TAG}] Saved {TAG}_timeseries.png")

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n[{TAG}] Summary:")
print(f"  Trajectory shape : {traj.shape}")
print(f"  Recordings h5    : recordings_{TAG}.h5  ({data.shape})")
print(f"  VISp4 nodes      : {len(stim_idx)}")
if len(stim_idx) > 0:
    print(f"  VISp4 E  t=0     : {data[0, stim_idx].mean():.4f}")
    print(f"  VISp4 E  t=500ms : {data[-1, stim_idx].mean():.4f}")
    delta = data[-1, stim_idx].mean() - data[0, stim_idx].mean()
    print(f"  dE (stim region) : {delta:+.4f}")
print(f"  Whole-pop E mean : {mean_E_all[0]:.4f} -> {mean_E_all[-1]:.4f}")
print(f"  Animation        : {TAG}_activity.gif")
