"""
Test WilsonCowan and JansenRit with visual cortex stimulation.
1s simulation, 1/4 population, LinearMean coupling, stream-only mode.
Outputs: recordings HDF5, time-series plot, spatial visualization.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesobrainsim.simulation import Simulation
from mesobrainsim.anatomy import Anatomy
from mesobrainsim.connectivity import Connectivity
from mesobrainsim.stimulation import ConstantStimulator
from mesobrainsim.measurement import MeasurementHook, Probe
from mesobrainsim.coupling import LinearMean
from mesobrainsim.utils import resolve_nodes

DATA = 'Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5'
T    = 1000.0  # ms (1 second)

# --- per-model configs ---
MODELS = {
    'WilsonCowan': dict(
        dt=0.1, rec=100,            # record every 100 steps -> 1 ms cadence
        model_kwargs=dict(tau_e=10.0, tau_i=20.0, P=0.5, c=0.0),
        coupling_kwargs=dict(c=0.5, var_index=0),   # couple on E
        stim_magnitude=2.0, stim_var=0,
        signal_var=0,                # probe E
        h5='recordings_wc_vis.h5',
    ),
    'JansenRit': dict(
        dt=0.05, rec=200,           # record every 200 steps -> 1 ms cadence
        model_kwargs=dict(p_mean=220.0, p_std=22.0, c=0.0),
        coupling_kwargs=dict(c=1.0, var_index=1),    # couple on y1
        stim_magnitude=20.0, stim_var=1,             # drive y4 (excitatory PSP input)
        signal_var=0,                # probe y0 (pyramidal PSP)
        h5='recordings_jr_vis.h5',
    ),
}

# regions to probe (Allen CCF leaf IDs)
VISp_IDS = [593, 821, 721, 778, 33, 305]   # primary visual (stimulated)
AUDp_IDS = [1047, 806, 862, 893, 656, 692] # primary auditory
MOp_IDS  = [648, 844, 882, 430]            # primary motor
RSP_IDS  = [687, 962, 943]                 # retrosplenial

REGIONS = {
    'VISp': VISp_IDS,
    'AUDp': AUDp_IDS,
    'MOp':  MOp_IDS,
    'RSP':  RSP_IDS,
}

# ---------------------------------------------------------------------------
# pre-build anatomy once (shared seed -> same node set)
# ---------------------------------------------------------------------------
anatomy      = Anatomy(DATA, subsample_fraction=0.25, seed=0)
connectivity = Connectivity(DATA, anatomy)
N            = len(anatomy)

print(f"N={N}")
print(f"Sample region_names: {anatomy.region_names[:5]}")
print(f"Sample region_ids:   {anatomy.region_ids[:5]}")

# resolve all region selectors
region_indices = {}
for label, sel in REGIONS.items():
    idx = resolve_nodes(anatomy, sel)
    region_indices[label] = idx
    print(f"  {label}: {len(idx)} nodes")


def run_model(model_name):
    cfg = MODELS[model_name]
    dt  = cfg['dt']
    rec = cfg['rec']
    n_recorded = int(T / dt) // rec + 1

    # stimulate VISp
    stim = ConstantStimulator(anatomy, selector=VISp_IDS,
                              magnitude=cfg['stim_magnitude'],
                              var_col=cfg['stim_var'],
                              t_start=100.0, t_end=600.0)

    # probes per region
    probes = []
    for label, idx in region_indices.items():
        if len(idx):
            probes.append(Probe(label, idx, var_index=cfg['signal_var']))
    # whole population
    probes.append(Probe('all', np.arange(N), var_index=cfg['signal_var']))

    mhook = MeasurementHook(probes, cfg['h5'], anatomy, connectivity.W, n_recorded)

    coupling = LinearMean(**cfg['coupling_kwargs'])

    sim = Simulation(
        data_path=DATA, subsample_fraction=0.25, seed=0,
        model=model_name, model_kwargs=cfg['model_kwargs'],
        solver='Heun', dt=dt, T=T,
        record_every=rec,
        stimulator=stim,
        measurement_hooks=[mhook],
        coupling_model=coupling,
        use_gpu=True,
        return_trajectory=False,
    )
    result = sim.run()
    assert result.trajectory is None, "expected stream-only"

    # verify
    with h5py.File(cfg['h5'], 'r') as f:
        times = f['metadata/times'][:]
        print(f"  [{model_name}] times: {times[0]:.1f} .. {times[-1]:.1f} ms, "
              f"n_recorded={len(times)}")
        for label in region_indices:
            if label in f['probes']:
                d = f[f'probes/{label}/data'][:]
                print(f"  [{model_name}] {label}: shape={d.shape}, "
                      f"mean start={d[0].mean():.4f}, mean end={d[-1].mean():.4f}")

    return cfg['h5']


def plot_timeseries(h5_wc, h5_jr):
    """Plot regional mean traces for both models."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {'VISp': 'red', 'AUDp': 'blue', 'MOp': 'green',
              'SSp': 'orange', 'RSP': 'purple'}

    for ax, (h5_path, model_name, ylabel) in zip(axes, [
        (h5_wc, 'WilsonCowan', 'E (excitatory)'),
        (h5_jr, 'JansenRit', 'y0 (pyramidal PSP)'),
    ]):
        with h5py.File(h5_path, 'r') as f:
            times = f['metadata/times'][:]
            for label, color in colors.items():
                if label in f['probes']:
                    data = f[f'probes/{label}/data'][:]
                    mean_trace = data.mean(axis=1)
                    ax.plot(times, mean_trace, color=color, lw=1.2,
                            label=f'{label} ({data.shape[1]} nodes)')

        ax.axvspan(100, 600, alpha=0.08, color='yellow', label='stim window')
        ax.set_ylabel(ylabel)
        ax.set_title(model_name)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')
    plt.tight_layout()
    out = 'scripts/wc_jr_timeseries.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved time-series: {out}")


def plot_spatial(h5_wc, h5_jr):
    """Spatial snapshots at peak stim (t~400 ms) for both models."""
    coords = anatomy.coords
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (h5_path, model_name) in zip(axes, [
        (h5_wc, 'WilsonCowan'),
        (h5_jr, 'JansenRit'),
    ]):
        with h5py.File(h5_path, 'r') as f:
            times = f['metadata/times'][:]
            data  = f['probes/all/data'][:]
        # snapshot near t=400 ms
        t_idx = np.argmin(np.abs(times - 400.0))
        signal = data[t_idx]

        sc = ax.scatter(coords[:, 0], coords[:, 2], c=signal,
                        cmap='turbo', s=1, alpha=0.6)
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Z (um)')
        ax.set_title(f'{model_name} @ t={times[t_idx]:.0f} ms')
        plt.colorbar(sc, ax=ax, shrink=0.7)

    plt.tight_layout()
    out = 'scripts/wc_jr_spatial.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Saved spatial: {out}")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("WilsonCowan — VISp stimulation, 1s")
    print("=" * 60)
    # lower self-excitation, raise inhibition -> excitable regime
    MODELS['WilsonCowan']['model_kwargs'] = dict(
        tau_e=10.0, tau_i=20.0,
        a_ee=10.0, a_ei=12.0, a_ie=8.0, a_ii=3.0,
        P=-0.5, Q=0.0, c=0.0,
    )
    MODELS['WilsonCowan']['coupling_kwargs'] = dict(c=0.1, var_index=0)
    MODELS['WilsonCowan']['stim_magnitude'] = 2.0
    h5_wc = run_model('WilsonCowan')

    print("\n" + "=" * 60)
    print("JansenRit — VISp stimulation, 1s")
    print("=" * 60)
    MODELS['JansenRit']['coupling_kwargs'] = dict(c=0.2, var_index=1)
    MODELS['JansenRit']['stim_magnitude'] = 10.0
    MODELS['JansenRit']['stim_var'] = 4   # inject into dy4 (excitatory PSP input)
    MODELS['JansenRit']['dt'] = 0.01      # small dt for JR stability (a=100)
    MODELS['JansenRit']['rec'] = 100      # keep ~1ms cadence
    h5_jr = run_model('JansenRit')

    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)
    plot_timeseries(h5_wc, h5_jr)
    plot_spatial(h5_wc, h5_jr)
    print("Done.")
