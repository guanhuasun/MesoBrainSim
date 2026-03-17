"""
Top-level orchestrator: wires Anatomy, Connectivity, model, solver, and Visualizer.
"""

from . import config
from .anatomy import Anatomy
from .connectivity import Connectivity
from .ephys import MODELS
from .solver import SOLVERS
from .viz import BrainVisualizer


class SimulationResult:
    """Holds outputs from a completed simulation run."""

    def __init__(self, trajectory, times, anatomy, connectivity):
        self.trajectory = trajectory    # (T, N, state_dim)
        self.times = times              # (T,)
        self.anatomy = anatomy
        self.connectivity = connectivity

    def visualize(self, t_index: int = -1, **kwargs):
        """Open a static 3D plot at a given time index."""
        viz = BrainVisualizer(self.anatomy.coords, self.trajectory, self.times)
        viz.plot_static(t_index=t_index, **kwargs)

    def animate(self, output_path: str = "brain_sim.gif", **kwargs):
        """Render an animation of signal propagation."""
        viz = BrainVisualizer(self.anatomy.coords, self.trajectory, self.times)
        viz.animate(output_path=output_path, **kwargs)


class Simulation:
    """
    Configures and runs a whole-brain simulation.

    Parameters
    ----------
    data_path : str
        Path to the HDF5 data file.
    n_nodes : int, optional
        Number of nodes to subsample (default: all).
    subsample_fraction : float, optional
        Alternative to n_nodes; fraction of total nodes to use.
    model : str or NeuralModel instance
        Model name (key in ephys.MODELS) or pre-instantiated model.
    model_kwargs : dict, optional
        Keyword arguments passed to the model constructor when model is a str.
    solver : str or solver instance
        Solver name ('Euler' or 'Heun') or pre-instantiated solver.
    dt : float
        Integration time step (ms or model-native units).
    T : float
        Total simulation duration.
    record_every : int
        Store state every this many steps.
    use_gpu : bool
        If True, use CuPy backend.
    seed : int
        Random seed for subsampling.

    Example
    -------
    >>> sim = Simulation(
    ...     data_path="data/Zhuang-ABCA-1-Isocortex_rho299_inh_local.h5",
    ...     n_nodes=500,
    ...     model="WilsonCowan",
    ...     solver="Heun",
    ...     dt=0.1,
    ...     T=200.0,
    ... )
    >>> result = sim.run()
    >>> result.visualize()
    """

    def __init__(
        self,
        data_path: str,
        n_nodes: int = None,
        subsample_fraction: float = None,
        model="WilsonCowan",
        model_kwargs: dict = None,
        solver="Heun",
        dt: float = 0.1,
        T: float = 200.0,
        record_every: int = 1,
        use_gpu: bool = False,
        seed: int = 0,
    ):
        self.data_path = data_path
        self.n_nodes = n_nodes
        self.subsample_fraction = subsample_fraction
        self.dt = dt
        self.T = T
        self.record_every = record_every
        self.seed = seed

        # Backend
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

    def run(self) -> SimulationResult:
        """Load data, build model, integrate, and return results."""
        print(f"[Simulation] Loading anatomy from '{self.data_path}'...")
        anatomy = Anatomy(
            self.data_path,
            n_nodes=self.n_nodes,
            subsample_fraction=self.subsample_fraction,
            seed=self.seed,
        )

        print("[Simulation] Loading connectivity...")
        connectivity = Connectivity(self.data_path, anatomy)

        print(f"[Simulation] Running {type(self.model).__name__} "
              f"with {type(self.solver).__name__} "
              f"(N={len(anatomy)}, dt={self.dt}, T={self.T})...")

        trajectory, times = self.solver.run(
            model=self.model,
            W=connectivity.W,
            T=self.T,
            dt=self.dt,
            record_every=self.record_every,
        )

        print(f"[Simulation] Done. Trajectory shape: {trajectory.shape}")
        return SimulationResult(trajectory, times, anatomy, connectivity)
