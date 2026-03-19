"""Hebbian and STDP weight update rules; in-place modification of W."""

import numpy as np
from . import config


class PlasticityHook:
    def __init__(self, update_every=1, w_max=None, w_min=None):
        self.update_every = update_every
        self.w_max = w_max
        self.w_min = w_min
        self._W = None
        self._is_sparse = False
        self._row_of = None  # xp array; only for sparse

    def attach(self, W, is_sparse):
        self._W = W
        self._is_sparse = is_sparse
        if is_sparse:
            xp = config.xp
            N = W.shape[0]
            # indptr may be CuPy; get CPU copy for np.diff/repeat
            indptr = W.indptr.get() if config.USE_GPU else np.asarray(W.indptr)
            row_of_cpu = np.repeat(np.arange(N, dtype=np.int32), np.diff(indptr))
            self._row_of = xp.asarray(row_of_cpu)

    def __call__(self, step, t, state):
        if step % self.update_every == 0:
            self._update(t, state)
            self._clip()

    def _update(self, t, state):
        raise NotImplementedError

    def _clip(self):
        W = self._W
        lo, hi = self.w_min, self.w_max
        if lo is None and hi is None:
            return
        xp = config.xp
        if self._is_sparse:
            if lo is not None:
                xp.maximum(W.data, lo, out=W.data)
            if hi is not None:
                xp.minimum(W.data, hi, out=W.data)
        else:
            xp.clip(W, lo, hi, out=W)


class HebbianPlasticity(PlasticityHook):
    """W += eta * outer(signal, signal); zero diagonal; clip."""

    def __init__(self, eta, signal_var=0, update_every=1, w_max=None, w_min=None):
        super().__init__(update_every, w_max, w_min)
        self.eta = eta
        self.signal_var = signal_var

    def _update(self, t, state):
        xp = config.xp
        W = self._W
        signal = state[:, self.signal_var]          # (N,)
        N = signal.shape[0]

        if self._is_sparse:
            W.data += self.eta * signal[self._row_of] * signal[W.indices]
        else:
            if config.USE_GPU:
                try:
                    import cupy as cp
                    dW = self.eta * cp.outer(signal, signal)
                except Exception:
                    dW = self.eta * xp.outer(signal, signal)
            else:
                dW = self.eta * xp.outer(signal, signal)
            dW[xp.arange(N), xp.arange(N)] = 0
            W += dW


class HomeostaticPlasticity(PlasticityHook):
    """Scale weights to keep mean activity near a target rate."""

    def __init__(self, target_rate, eta=0.001, signal_var=0,
                 update_every=10, w_max=None, w_min=None):
        super().__init__(update_every, w_max, w_min)
        self.target_rate = target_rate
        self.eta = eta
        self.signal_var = signal_var

    def _update(self, t, state):
        xp = config.xp
        W = self._W
        activity = state[:, self.signal_var]
        mean_act = float(xp.mean(activity))
        scale = 1.0 + self.eta * (self.target_rate - mean_act)
        if self._is_sparse:
            W.data *= xp.float32(scale)
        else:
            W *= xp.float32(scale)


class STDPPlasticity(PlasticityHook):
    """
    Spike-timing dependent plasticity with exponential trace variables.
    Trace updates and weight updates both happen at `update_every` cadence.
    """

    def __init__(self, tau_plus, tau_minus, A_plus, A_minus,
                 threshold=0.5, signal_var=0, is_spiking=True,
                 dt=0.1, update_every=1, w_max=None, w_min=None):
        super().__init__(update_every, w_max, w_min)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.threshold = threshold
        self.signal_var = signal_var
        self.is_spiking = is_spiking
        self.dt = dt
        self._x_pre = None
        self._x_post = None

    def attach(self, W, is_sparse):
        super().attach(W, is_sparse)
        xp = config.xp
        N = W.shape[0]
        self._x_pre = xp.zeros(N, dtype=xp.float32)
        self._x_post = xp.zeros(N, dtype=xp.float32)

    def _update(self, t, state):
        xp = config.xp
        W = self._W
        activity = state[:, self.signal_var]
        N = activity.shape[0]

        if self.is_spiking:
            spikes = (activity >= self.threshold).astype(xp.float32)
        else:
            spikes = activity.astype(xp.float32)

        # decay traces and add spike contribution
        eff_dt = self.dt * self.update_every
        self._x_pre = self._x_pre * xp.exp(xp.array(-eff_dt / self.tau_plus, dtype=xp.float32)) + spikes
        self._x_post = self._x_post * xp.exp(xp.array(-eff_dt / self.tau_minus, dtype=xp.float32)) + spikes

        if self._is_sparse:
            W.data += (
                self.A_plus  * self._x_pre[self._row_of]  * spikes[W.indices]
              - self.A_minus * spikes[self._row_of]        * self._x_post[W.indices]
            )
        else:
            dW = (
                self.A_plus  * xp.outer(self._x_pre, spikes)
              - self.A_minus * xp.outer(spikes, self._x_post)
            )
            dW[xp.arange(N), xp.arange(N)] = 0
            W += dW
