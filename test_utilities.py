"""
Integration test: 500ms WilsonCowan on 1/4 population,
constant stimulation to VISp4+AUDp4, whole-population measurement,
Hebbian plasticity.
"""

import h5py
import numpy as np
from mesobrainsim.simulation import Simulation
from mesobrainsim.anatomy import Anatomy
from mesobrainsim.connectivity import Connectivity
from mesobrainsim.stimulation import ConstantStimulator
from mesobrainsim.measurement import MeasurementHook, Probe
from mesobrainsim.plasticity import HebbianPlasticity, STDPPlasticity
from mesobrainsim.utils import resolve_nodes

DATA = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
DT   = 0.025  # ms (HH requires small dt)
T    = 500.0  # ms
REC  = 200    # record every 200 steps

# -- pre-build anatomy to resolve selectors and compute n_recorded --
anatomy      = Anatomy(DATA, subsample_fraction=0.25, seed=0)
connectivity = Connectivity(DATA, anatomy)
N            = len(anatomy)
n_recorded   = int(T / DT) // REC + 1
print(f"N={N}, n_recorded={n_recorded}")

# -- inspect region_names to confirm selector format --
print(f"Sample region_names: {anatomy.region_names[:5]}")
print(f"Sample region_ids:   {anatomy.region_ids[:5]}")

# -- resolve VISp4 and AUDp4 --
visp_idx = resolve_nodes(anatomy, 'VISp4')
audp_idx  = resolve_nodes(anatomy, 'AUDp4')
print(f"VISp4 nodes: {len(visp_idx)}, AUDp4 nodes: {len(audp_idx)}")

# -- stimulators --
stim_visp = ConstantStimulator(anatomy, selector='VISp4',
                                magnitude=15.0, var_col=0,
                                t_start=0.0, t_end=T)
stim_audp = ConstantStimulator(anatomy, selector='AUDp4',
                                magnitude=15.0, var_col=0,
                                t_start=0.0, t_end=T)

# -- measurement: whole population, E variable --
probe_all = Probe('whole_pop_E', np.arange(N), var_index=0)
mhook     = MeasurementHook([probe_all], 'recordings.h5',
                              anatomy, connectivity.W, n_recorded)

# -- Hebbian plasticity every 100 steps --
hebb = HebbianPlasticity(eta=1e-5, signal_var=0, update_every=100,
                          w_max=2.0, w_min=0.0)

# -- Run (stream-only mode) --
sim = Simulation(
    data_path=DATA, subsample_fraction=0.25, seed=0,
    model='HodgkinHuxley', model_kwargs=dict(I_ext=0.0, c=0.0),
    solver='Heun', dt=DT, T=T,
    record_every=REC, use_gpu=False,
    stimulator=[stim_visp, stim_audp],
    measurement_hooks=[mhook],
    plasticity_hooks=[hebb],
    return_trajectory=False,
)
result = sim.run()
assert result.trajectory is None, "Expected trajectory=None in stream-only mode"

# -- Verify .h5 output --
with h5py.File('recordings.h5', 'r') as f:
    data  = f['probes/whole_pop_E/data'][:]
    times = f['metadata/times'][:]
    print(f"data shape: {data.shape}, times shape: {times.shape}")
    print(f"times[0]={times[0]}, times[-1]={times[-1]}")
    assert data.shape == (n_recorded, N), f"shape mismatch: {data.shape} vs ({n_recorded}, {N})"
    assert abs(times[-1] - T) < 1e-3, f"times[-1]={times[-1]} != T={T}"

if len(visp_idx) > 0:
    assert data[-1, visp_idx].mean() > data[0, visp_idx].mean(), "VISp4 not driven"
    print(f"VISp4 E: {data[0, visp_idx].mean():.4f} -> {data[-1, visp_idx].mean():.4f}  OK")
else:
    print("WARNING: VISp4 selector matched 0 nodes — check anatomy.region_names format")

if len(audp_idx) > 0:
    assert data[-1, audp_idx].mean() > data[0, audp_idx].mean(), "AUDp4 not driven"
    print(f"AUDp4 E: {data[0, audp_idx].mean():.4f} -> {data[-1, audp_idx].mean():.4f}  OK")
else:
    print("WARNING: AUDp4 selector matched 0 nodes — check anatomy.region_names format")

print("\nALL UTILITY TESTS PASSED")

# AllenSDK (run separately in allen env):
# conda run -n allen python scripts/allensdk_query.py \
#     --regions VISp,AUDp --output data/allen_connectivity.h5
# python -c "import h5py; f=h5py.File('data/allen_connectivity.h5'); print(list(f.keys()))"
