"""External current injection: constant or Poisson spike trains."""

import numpy as np
from . import config
from .utils import resolve_nodes


class StimulatorHook:
    """Base stimulator; subclasses implement __call__."""

    def __init__(self, anatomy, selector, var_col, t_start, t_end):
        self.node_indices = resolve_nodes(anatomy, selector)
        self.var_col = var_col
        self.t_start = t_start
        self.t_end = t_end
        self._buffer = None
        self._mask = None

    def prepare(self, N):
        xp = config.xp
        self._buffer = xp.zeros(N, dtype=xp.float32)
        self._mask = xp.zeros(N, dtype=bool)
        if len(self.node_indices):
            self._mask[self.node_indices] = True

    def get_shim(self, model, var_col=None):
        """Return dfdt closure that adds self._buffer to d[:, var_col]."""
        vc = self.var_col
        buf = self._buffer
        def _shim(state, t, W, coupling=None):
            d = model.dfdt(state, t, W, coupling=coupling)
            d[:, vc] = d[:, vc] + buf
            return d
        return _shim

    def __call__(self, t):
        raise NotImplementedError


class ConstantStimulator(StimulatorHook):
    """Inject constant current into selected nodes during [t_start, t_end]."""

    def __init__(self, anatomy, selector, magnitude, var_col=0,
                 t_start=0.0, t_end=float('inf')):
        super().__init__(anatomy, selector, var_col, t_start, t_end)
        self.magnitude = magnitude

    def __call__(self, t):
        xp = config.xp
        self._buffer[:] = 0
        if self.t_start <= t <= self.t_end:
            self._buffer[self._mask] = self.magnitude
        return self._buffer


class PoissonStimulator(StimulatorHook):
    """Bernoulli spike events at rate_hz; no Python loop over nodes."""

    def __init__(self, anatomy, selector, rate_hz, magnitude, dt, var_col=0,
                 t_start=0.0, t_end=float('inf'), seed=None):
        super().__init__(anatomy, selector, var_col, t_start, t_end)
        self.rate_hz = rate_hz
        self.magnitude = magnitude
        self.dt = dt
        self._rng = np.random.default_rng(seed)

    def __call__(self, t):
        xp = config.xp
        self._buffer[:] = 0
        if self.t_start <= t <= self.t_end and len(self.node_indices):
            k = len(self.node_indices)
            p = self.rate_hz * self.dt * 1e-3
            spikes = self._rng.random(k) < p
            self._buffer[self.node_indices] = self.magnitude * xp.asarray(
                spikes.astype(np.float32)
            )
        return self._buffer


class MultiStimulator:
    """Sum buffers from multiple StimulatorHook instances."""

    def __init__(self, stimulators):
        self.stimulators = stimulators

    def prepare(self, N):
        for s in self.stimulators:
            s.prepare(N)

    def get_shim(self, model, var_col=None):
        stims = self.stimulators
        def _shim(state, t, W, coupling=None):
            d = model.dfdt(state, t, W, coupling=coupling)
            for s in stims:
                d[:, s.var_col] = d[:, s.var_col] + s._buffer
            return d
        return _shim

    def __call__(self, t):
        for s in self.stimulators:
            s(t)
