"""Coupling models: compute inter-node interaction terms from state and W."""

import numpy as np
from . import config


class CouplingModel:
    """Base class for coupling models."""

    def __init__(self, c=1.0, var_index=0):
        self.c = c
        self.var_index = var_index

    def _degree(self, W):
        xp = config.xp
        s = W.sum(axis=1)
        if hasattr(s, 'A1'):       # sparse matrix .sum() returns np.matrix
            s = s.A1
        elif hasattr(s, 'get'):    # cupy array
            s = s
        s = xp.asarray(s, dtype=xp.float32).ravel()
        return xp.maximum(s, 1.0)

    def compute(self, state, W):
        """Return coupling term, shape (N,)."""
        raise NotImplementedError


class LinearMean(CouplingModel):
    """c * W @ x / deg — degree-normalized mean-field coupling."""

    def compute(self, state, W):
        xp = config.xp
        x = state[:, self.var_index]
        deg = self._degree(W)
        return self.c * xp.asarray(W @ x).ravel() / deg


class LinearDiffusive(CouplingModel):
    """c * (W @ x - deg * x) / deg — diffusive coupling (mean neighbor - self)."""

    def compute(self, state, W):
        xp = config.xp
        x = state[:, self.var_index]
        deg = self._degree(W)
        Wx = xp.asarray(W @ x).ravel()
        return self.c * (Wx - deg * x) / deg


class SigmoidalCoupling(CouplingModel):
    """c * W @ sigmoid(x) / deg — nonlinear sigmoidal coupling."""

    def __init__(self, c=1.0, var_index=0, gain=1.0, threshold=0.0):
        super().__init__(c, var_index)
        self.gain = gain
        self.threshold = threshold

    def compute(self, state, W):
        xp = config.xp
        x = state[:, self.var_index]
        sig = 1.0 / (1.0 + xp.exp(-self.gain * (x - self.threshold)))
        deg = self._degree(W)
        return self.c * xp.asarray(W @ sig).ravel() / deg


class DelayCoupling(CouplingModel):
    """Distance-dependent conduction delays with ring buffer."""

    def __init__(self, c=1.0, var_index=0, velocity=1.0, dt=0.1):
        super().__init__(c, var_index)
        self.velocity = velocity
        self.dt = dt
        self._buffer = None
        self._delay_steps = None
        self._max_delay = 0
        self._ptr = 0
        self._initialized = False

    def initialize(self, N, D):
        """Call before simulation with distance matrix D (N,N)."""
        xp = config.xp
        delays = D / self.velocity
        self._delay_steps = np.clip((delays / self.dt).astype(np.int32), 0, None)
        self._max_delay = int(self._delay_steps.max()) + 1
        self._buffer = xp.zeros((self._max_delay, N), dtype=xp.float32)
        self._ptr = 0
        self._initialized = True

    def compute(self, state, W):
        xp = config.xp
        x = state[:, self.var_index]

        if not self._initialized:
            # fallback to instantaneous mean-field
            deg = self._degree(W)
            return self.c * xp.asarray(W @ x).ravel() / deg

        # store current
        buf_idx = self._ptr % self._max_delay
        self._buffer[buf_idx] = x
        self._ptr += 1

        # delayed readout — per-connection delay
        N = x.shape[0]
        coupled = xp.zeros(N, dtype=xp.float32)
        deg = self._degree(W)

        # vectorized: for each delay d, mask connections with that delay
        for d in range(self._max_delay):
            read_idx = (self._ptr - 1 - d) % self._max_delay
            x_delayed = self._buffer[read_idx]
            # mask W to only connections with delay == d
            mask = (self._delay_steps == d)
            W_d = W * xp.asarray(mask.astype(np.float32))
            coupled += xp.asarray(W_d @ x_delayed).ravel()

        return self.c * coupled / deg


COUPLING_MODELS = {
    "LinearMean": LinearMean,
    "LinearDiffusive": LinearDiffusive,
    "Sigmoidal": SigmoidalCoupling,
    "Delay": DelayCoupling,
}
