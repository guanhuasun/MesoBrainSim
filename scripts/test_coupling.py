"""
Coupling strength sweep: HH degree-normalized gap junction, 5 c values.
Pre-builds anatomy/connectivity once; runs solver directly to avoid reload.
Propagation metric: fraction of non-stim nodes deviating >0.5 mV from rest.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import h5py

from mesobrainsim.anatomy import Anatomy
from mesobrainsim.connectivity import Connectivity
from mesobrainsim.ephys import HodgkinHuxley
from mesobrainsim.solver import HeunSolver
from mesobrainsim.stimulation import ConstantStimulator
from mesobrainsim.measurement import MeasurementHook, Probe
from mesobrainsim.utils import resolve_nodes

DATA     = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
FRAC     = 0.05    # ~14890 nodes (avg degree ~14)
SEED     = 0
DT       = 0.025   # ms
T        = 100.0   # ms
REC      = 40      # -> ~101 frames
MAG      = 15.0    # uA/cm^2
STIM_ID  = 593     # VISp4
# c scales: mean_neighbor_V - V_i ~ 7.5mV per firing neighbor/14 total
# c=2 -> coupled~15 uA/cm^2 from 1 VISp4 neighbor -> fires
C_SWEEP  = [0.0, 0.5, 1.0, 2.0, 5.0]
THRESH   = 0.5     # mV deviation from rest to count as "activated"

# ── build once ───────────────────────────────────────────────────────────────
print("Building anatomy/connectivity...")
anatomy      = Anatomy(DATA, subsample_fraction=FRAC, seed=SEED)
connectivity = Connectivity(DATA, anatomy)
N            = len(anatomy)
n_recorded   = int(T / DT) // REC + 1
print(f"N={N}, n_recorded={n_recorded}")

stim_idx = resolve_nodes(anatomy, STIM_ID)
if len(stim_idx) == 0:
    stim_idx = resolve_nodes(anatomy, 'VISp4')
print(f"VISp4 nodes: {len(stim_idx)}")

non_stim_mask = np.ones(N, dtype=bool)
if len(stim_idx):
    non_stim_mask[stim_idx] = False

solver  = HeunSolver()
results = {}

print(f"\n{'c':<6}  {'stim_dV':>10}  {'non_stim_dV':>12}  {'frac_act':>10}  propagates")
print("-" * 60)

# ── sweep ────────────────────────────────────────────────────────────────────
for c in C_SWEEP:
    tag = f"c{c}".replace('.', 'p')
    h5  = f"coupling_sweep_{tag}.h5"

    model = HodgkinHuxley(I_ext=0.0, c=c)
    stim  = ConstantStimulator(anatomy, selector=STIM_ID,
                                magnitude=MAG, var_col=0,
                                t_start=0.0, t_end=T)
    probe = Probe('V', np.arange(N), var_index=0)
    mhook = MeasurementHook([probe], h5, anatomy, connectivity.W, n_recorded)

    solver.run(model, connectivity.W, T, DT,
               record_every=REC, hooks=[mhook],
               stimulator=stim, return_trajectory=False)

    with h5py.File(h5, 'r') as f:
        data = f['probes/V/data'][:]          # (n_recorded, N)

    v_rest      = -65.0
    dv_stim     = data[-1, stim_idx].mean() - v_rest   if len(stim_idx) else float('nan')
    dv_non      = data[-1, non_stim_mask].mean() - v_rest
    deviation   = np.abs(data[-1, non_stim_mask] - v_rest)
    frac_act    = float((deviation > THRESH).mean())
    propagated  = frac_act > 0.01   # >1% of non-stim nodes deviated

    has_nan  = bool(np.isnan(data).any())
    results[c] = dict(dv_stim=dv_stim, dv_non=dv_non,
                      frac_act=frac_act, propagated=propagated, nan=has_nan)
    nan_note = '  [NaN: reduce dt]' if has_nan else ''
    dv_s_str = f"{dv_stim:>+10.3f}" if not np.isnan(dv_stim) else f"{'nan':>10}"
    dv_n_str = f"{dv_non:>+12.4f}"  if not np.isnan(dv_non)  else f"{'nan':>12}"
    print(f"{c:<6}  {dv_s_str}  {dv_n_str}  {frac_act:>10.4f}"
          f"  {'YES' if propagated else 'no'}{nan_note}")

# ── assertions ────────────────────────────────────────────────────────────────
print()
assert not results[0.0]['propagated'],  "c=0 should NOT propagate"
assert results[5.0]['propagated'],      "c=5.0 should propagate"
# monotonic: higher c -> at least as much propagation
assert results[5.0]['frac_act'] >= results[2.0]['frac_act'], \
    "c=5.0 should activate at least as many nodes as c=2.0"
print("ALL COUPLING SWEEP ASSERTIONS PASSED")
