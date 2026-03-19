"""
JansenRit visual cortex stimulation with coupling sweep.
Stim VISp for first 100ms, run 1s total, sweep coupling strength.
Outputs: recordings HDF5, time-series plot, 3D animated GIF per coupling.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mesobrainsim.anatomy import Anatomy
from mesobrainsim.connectivity import Connectivity
from mesobrainsim.stimulation import ConstantStimulator
from mesobrainsim.measurement import MeasurementHook, Probe
from mesobrainsim.simulation import Simulation
from mesobrainsim.coupling import LinearMean
from mesobrainsim.viz import BrainVisualizer
from mesobrainsim.utils import resolve_nodes

# ── config ──────────────────────────────────────────────────────────────────
DATA = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
FRAC = 0.25
SEED = 0
DT   = 0.01     # ms — JR stiff (a=100), needs dt < 0.02 for Heun stability
T    = 500.0    # ms
REC  = 100      # record every 100 steps -> 1ms cadence

# VISp leaf IDs (Allen CCF)
VISp_IDS = [593, 821, 721, 778, 33, 305]
# other regions for comparison
AUDp_IDS = [1047, 806, 862, 893, 656, 692]
MOp_IDS  = [648, 844, 882, 430]
RSP_IDS  = [687, 962, 943]

STIM_MAG = 200.0  # injected into dy4
STIM_VAR = 4      # dy[4] = A*a*(p + ... + coupled) - ...
PROBE_VAR = 1     # probe y1 (excitatory PSP, directly affected by stim + coupling)
# no stochastic drive — stimulation is the only input
JR_KWARGS = dict(p_mean=0.0, p_std=0.0, c=0.0)

C_VALUES = [0.0, 0.5, 2.0]  # coupling sweep

# ── pre-build anatomy (shared across runs) ──────────────────────────────────
print("[Setup] Loading anatomy + connectivity...")
anatomy      = Anatomy(DATA, subsample_fraction=FRAC, seed=SEED)
connectivity = Connectivity(DATA, anatomy)
N = len(anatomy)

regions = {
    'VISp': resolve_nodes(anatomy, VISp_IDS),
    'AUDp': resolve_nodes(anatomy, AUDp_IDS),
    'MOp':  resolve_nodes(anatomy, MOp_IDS),
    'RSP':  resolve_nodes(anatomy, RSP_IDS),
}
for label, idx in regions.items():
    print(f"  {label}: {len(idx)} nodes")

n_steps    = int(T / DT)
n_recorded = n_steps // REC + 1


def run_jr(c_val):
    tag = f"jr_c{c_val:.1f}".replace('.', 'p')
    h5_path = f'recordings_{tag}.h5'
    print(f"\n{'='*60}")
    print(f"JansenRit c={c_val} — VISp stim 0-100ms, T=1s")
    print(f"{'='*60}")

    # stim VISp first 100ms only
    stim = ConstantStimulator(
        anatomy, selector=VISp_IDS,
        magnitude=STIM_MAG, var_col=STIM_VAR,
        t_start=0.0, t_end=100.0,
    )

    # probes: per-region + whole pop
    probes = [Probe(label, idx, var_index=PROBE_VAR) for label, idx in regions.items() if len(idx)]
    probes.append(Probe('all', np.arange(N), var_index=PROBE_VAR))
    mhook = MeasurementHook(probes, h5_path, anatomy, connectivity.W, n_recorded)

    coupling = LinearMean(c=c_val, var_index=1) if c_val > 0 else None

    sim = Simulation(
        data_path=DATA, subsample_fraction=FRAC, seed=SEED,
        model='JansenRit',
        model_kwargs=JR_KWARGS,
        solver='Heun', dt=DT, T=T,
        record_every=REC, use_gpu=True,
        stimulator=stim,
        measurement_hooks=[mhook],
        coupling_model=coupling,
        return_trajectory=True,
    )
    result = sim.run()

    # verify
    with h5py.File(h5_path, 'r') as f:
        times = f['metadata/times'][:]
        for label in regions:
            if label in f['probes']:
                d = f[f'probes/{label}/data'][:]
                print(f"  {label}: mean start={d[:5].mean():.4f}, "
                      f"mean end={d[-5:].mean():.4f}")

    return result, h5_path, tag


# ── run sweep ───────────────────────────────────────────────────────────────
results = {}
for c_val in C_VALUES:
    result, h5_path, tag = run_jr(c_val)
    results[c_val] = (result, h5_path, tag)

# ── time-series comparison plot ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("Generating plots")
print(f"{'='*60}")

colors = {'VISp': 'red', 'AUDp': 'blue', 'MOp': 'green', 'RSP': 'purple'}
fig, axes = plt.subplots(len(C_VALUES), 1, figsize=(12, 3.5 * len(C_VALUES)), sharex=True)
if len(C_VALUES) == 1:
    axes = [axes]

for ax, c_val in zip(axes, C_VALUES):
    _, h5_path, tag = results[c_val]
    with h5py.File(h5_path, 'r') as f:
        times = f['metadata/times'][:]
        for label, color in colors.items():
            if label in f['probes']:
                data = f[f'probes/{label}/data'][:]
                mean_trace = data.mean(axis=1)
                ax.plot(times, mean_trace, color=color, lw=1.2,
                        label=f'{label} ({data.shape[1]})')

    ax.axvspan(0, 100, alpha=0.1, color='yellow', label='stim')
    ax.set_ylabel('y1 (excitatory PSP)')
    ax.set_title(f'JansenRit — coupling c={c_val}')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ms)')
plt.tight_layout()
out_ts = 'scripts/jr_coupling_timeseries.png'
plt.savefig(out_ts, dpi=150)
plt.close()
print(f"[Plot] Saved {out_ts}")

# ── 3D animated GIFs ────────────────────────────────────────────────────────
for c_val in C_VALUES:
    result, _, tag = results[c_val]
    traj  = result.trajectory   # (T_rec, N, 6)
    times = result.times

    # use y1-y2 (PSP output of pyramidal cells) for visualization
    if hasattr(traj, 'get'):
        traj_np = traj.get()
        times_np = times.get()
    else:
        traj_np = np.asarray(traj)
        times_np = np.asarray(times)

    # extract y1 for BrainVisualizer (it reads [:,:,0])
    traj_viz = traj_np[:, :, PROBE_VAR:PROBE_VAR+1]  # shape (T, N, 1)

    gif_path = f'scripts/jr_{tag}_activity.gif'
    print(f"[Viz] Rendering {gif_path} ...")
    viz = BrainVisualizer(anatomy.coords, traj_viz, times_np)
    viz.animate(output_path=gif_path, cmap='turbo', point_size=3.0,
                framerate=15, step=2)

print("\nDone.")
