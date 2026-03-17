"""Top-level orchestrator: wires Anatomy, Connectivity, model, solver, and Visualizer."""

from . import config
from .anatomy import Anatomy
from .connectivity import Connectivity
from .ephys import MODELS
from .solver import SOLVERS
from .viz import BrainVisualizer
from .stimulation import MultiStimulator


class SimulationResult:
    def __init__(self, trajectory, times, anatomy, connectivity, measurement_hooks=None):
        self.trajectory = trajectory          # (T, N, state_dim) or None
        self.times = times                    # (T,) or None
        self.anatomy = anatomy
        self.connectivity = connectivity
        self.measurement_hooks = measurement_hooks or []

    def load_probe(self, probe_name, h5_path):
        """Return (times, data) arrays for a named probe from a recordings .h5."""
        import h5py, numpy as np
        with h5py.File(h5_path, 'r') as f:
            times = f['metadata/times'][:]
            data = f[f'probes/{probe_name}/data'][:]
        return times, data

    def visualize(self, t_index=-1, **kwargs):
        if self.trajectory is None:
            raise RuntimeError("trajectory=None (stream-only mode); use load_probe to read data.")
        viz = BrainVisualizer(self.anatomy.coords, self.trajectory, self.times)
        viz.plot_static(t_index=t_index, **kwargs)

    def animate(self, output_path='brain_sim.gif', **kwargs):
        if self.trajectory is None:
            raise RuntimeError("trajectory=None (stream-only mode); use load_probe to read data.")
        viz = BrainVisualizer(self.anatomy.coords, self.trajectory, self.times)
        viz.animate(output_path=output_path, **kwargs)


class Simulation:
    def __init__(
        self,
        data_path: str,
        n_nodes: int = None,
        subsample_fraction: float = None,
        model='WilsonCowan',
        model_kwargs: dict = None,
        solver='Heun',
        dt: float = 0.1,
        T: float = 200.0,
        record_every: int = 1,
        use_gpu: bool = False,
        seed: int = 0,
        # new optional hooks
        stimulator=None,
        measurement_hooks=None,
        plasticity_hooks=None,
        allen_h5_path=None,
        return_trajectory=True,
    ):
        self.data_path = data_path
        self.n_nodes = n_nodes
        self.subsample_fraction = subsample_fraction
        self.dt = dt
        self.T = T
        self.record_every = record_every
        self.seed = seed
        self.allen_h5_path = allen_h5_path
        self.return_trajectory = return_trajectory
        self.measurement_hooks = measurement_hooks or []
        self.plasticity_hooks = plasticity_hooks or []

        config.set_backend(use_gpu)

        # Model
        if isinstance(model, str):
            if model not in MODELS:
                raise ValueError(f"Unknown model '{model}'. Available: {list(MODELS.keys())}")
            self.model = MODELS[model](**(model_kwargs or {}))
        else:
            self.model = model

        # Solver
        if isinstance(solver, str):
            if solver not in SOLVERS:
                raise ValueError(f"Unknown solver '{solver}'. Available: {list(SOLVERS.keys())}")
            self.solver = SOLVERS[solver]()
        else:
            self.solver = solver

        # Stimulator: list -> MultiStimulator
        if isinstance(stimulator, list):
            self.stimulator = MultiStimulator(stimulator)
        else:
            self.stimulator = stimulator

    def run(self) -> SimulationResult:
        print(f"[Simulation] Loading anatomy from '{self.data_path}'...")
        anatomy = Anatomy(
            self.data_path,
            n_nodes=self.n_nodes,
            subsample_fraction=self.subsample_fraction,
            seed=self.seed,
        )

        print("[Simulation] Loading connectivity...")
        connectivity = Connectivity(self.data_path, anatomy)

        if self.allen_h5_path:
            print(f"[Simulation] Loading Allen weights from '{self.allen_h5_path}'...")
            connectivity.load_allen_weights(self.allen_h5_path, anatomy)

        print(f"[Simulation] Running {type(self.model).__name__} "
              f"with {type(self.solver).__name__} "
              f"(N={len(anatomy)}, dt={self.dt}, T={self.T})...")

        trajectory, times = self.solver.run(
            model=self.model,
            W=connectivity.W,
            T=self.T,
            dt=self.dt,
            record_every=self.record_every,
            hooks=self.measurement_hooks,
            stimulator=self.stimulator,
            plasticity_hooks=self.plasticity_hooks,
            return_trajectory=self.return_trajectory,
        )

        if self.return_trajectory:
            print(f"[Simulation] Done. Trajectory shape: {trajectory.shape}")
        else:
            print("[Simulation] Done. Stream-only mode (trajectory not retained).")

        return SimulationResult(
            trajectory, times, anatomy, connectivity,
            measurement_hooks=self.measurement_hooks,
        )
