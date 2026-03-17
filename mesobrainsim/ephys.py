"""
Electrophysiology module: neural mass and spiking neuron models.

All models implement the NeuralModel interface:
    dfdt(state, t, W) -> d_state/dt

State arrays have shape (N, state_dim) where N is the number of nodes.
Parameters can be scalars (broadcast over all nodes) or arrays of shape (N,).
"""

import numpy as np
from . import config


class NeuralModel:
    """Abstract base class for all neural models."""

    #: Number of state variables per node
    state_dim: int = 1

    def initial_state(self, N: int):
        """Return a zero initial state of shape (N, state_dim)."""
        xp = config.xp
        return xp.zeros((N, self.state_dim), dtype=xp.float32)

    def dfdt(self, state, t: float, W):
        """
        Compute time derivative of state.

        Parameters
        ----------
        state : xp.ndarray, shape (N, state_dim)
        t     : float, current time
        W     : xp.ndarray, shape (N, N), weight matrix

        Returns
        -------
        xp.ndarray, shape (N, state_dim)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Wilson-Cowan
# ---------------------------------------------------------------------------

class WilsonCowan(NeuralModel):
    """
    Wilson-Cowan neural field model with excitatory (E) and inhibitory (I)
    populations.

    State: [E, I], shape (N, 2)

    dE/dt = (-E + S(a_ee*E - a_ei*I + P + c*W@E)) / tau_e
    dI/dt = (-I + S(a_ie*E - a_ii*I + Q))         / tau_i
    S(x)  = 1 / (1 + exp(-x))
    """

    state_dim = 2

    def __init__(self, tau_e=10.0, tau_i=20.0,
                 a_ee=12.0, a_ei=4.0, a_ie=13.0, a_ii=11.0,
                 P=0.5, Q=0.0, c=0.1):
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.a_ee = a_ee
        self.a_ei = a_ei
        self.a_ie = a_ie
        self.a_ii = a_ii
        self.P = P
        self.Q = Q
        self.c = c

    @staticmethod
    def _sigmoid(x):
        xp = config.xp
        return 1.0 / (1.0 + xp.exp(-x))

    def dfdt(self, state, t, W):
        xp = config.xp
        E = state[:, 0]
        I = state[:, 1]

        coupled = self.c * xp.asarray(W @ E).ravel()

        dE = (-E + self._sigmoid(self.a_ee * E - self.a_ei * I + self.P + coupled)) / self.tau_e
        dI = (-I + self._sigmoid(self.a_ie * E - self.a_ii * I + self.Q)) / self.tau_i

        return xp.stack([dE, dI], axis=1)

    def initial_state(self, N):
        xp = config.xp
        rng = np.random.default_rng(42)
        s = xp.array(rng.uniform(0.0, 0.1, size=(N, 2)).astype(np.float32))
        return s


# ---------------------------------------------------------------------------
# Jansen-Rit
# ---------------------------------------------------------------------------

class JansenRit(NeuralModel):
    """
    Jansen-Rit cortical column model.

    State: [y0, y1, y2, y3, y4, y5], shape (N, 6)

    Models three neural populations: pyramidal cells, excitatory interneurons,
    and inhibitory interneurons.
    """

    state_dim = 6

    def __init__(self, A=3.25, B=22.0, a=100.0, b=50.0,
                 v0=6.0, e0=2.5, r=0.56,
                 C1=135.0, C2=108.0, C3=33.75, C4=33.75,
                 p_mean=220.0, p_std=22.0, c=1.0):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.v0 = v0
        self.e0 = e0
        self.r = r
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.p_mean = p_mean
        self.p_std = p_std
        self.c = c

    def _sigm(self, v):
        xp = config.xp
        return (2.0 * self.e0) / (1.0 + xp.exp(self.r * (self.v0 - v)))

    def dfdt(self, state, t, W):
        xp = config.xp
        y = [state[:, i] for i in range(6)]

        rng = np.random.default_rng(int(t * 1000) % (2**31))
        N = state.shape[0]
        p = float(self.p_mean) + float(self.p_std) * xp.array(
            rng.standard_normal(N).astype(np.float32)
        )

        coupled = self.c * xp.asarray(W @ y[1]).ravel()

        dy = [None] * 6
        dy[0] = y[3]
        dy[3] = (self.A * self.a * self._sigm(y[1] - y[2])
                 - 2.0 * self.a * y[3]
                 - self.a ** 2 * y[0])
        dy[1] = y[4]
        dy[4] = (self.A * self.a * (p + self.C2 * self._sigm(self.C1 * y[0]) + coupled)
                 - 2.0 * self.a * y[4]
                 - self.a ** 2 * y[1])
        dy[2] = y[5]
        dy[5] = (self.B * self.b * self.C4 * self._sigm(self.C3 * y[0])
                 - 2.0 * self.b * y[5]
                 - self.b ** 2 * y[2])

        return xp.stack(dy, axis=1)


# ---------------------------------------------------------------------------
# Leaky Integrate-and-Fire
# ---------------------------------------------------------------------------

class IntegrateAndFire(NeuralModel):
    """
    Leaky Integrate-and-Fire (LIF) neuron.

    State: [V], shape (N, 1)

    tau * dV/dt = -(V - V_rest) + R * (I_ext + c * W @ spikes)

    Spike detection and reset are handled inside dfdt via soft-reset
    (a smooth approximation) to remain compatible with vectorized ODE solvers.
    For hard reset semantics, use the Simulation class to apply threshold
    crossings between solver steps.
    """

    state_dim = 1

    def __init__(self, tau=20.0, V_rest=-65.0, V_thresh=-50.0,
                 V_reset=-65.0, R=10.0, I_ext=2.0, c=0.5):
        self.tau = tau
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.R = R
        self.I_ext = I_ext
        self.c = c

    def dfdt(self, state, t, W):
        xp = config.xp
        V = state[:, 0]

        spikes = (V >= self.V_thresh).astype(xp.float32)
        coupled = self.c * xp.asarray(W @ spikes).ravel()

        dV = (-(V - self.V_rest) + self.R * (self.I_ext + coupled)) / self.tau

        # Soft reset: pull V back after threshold crossing
        dV = dV - (V - self.V_reset) * spikes

        return xp.expand_dims(dV, axis=1)

    def initial_state(self, N):
        xp = config.xp
        return xp.full((N, 1), self.V_rest, dtype=xp.float32)


# ---------------------------------------------------------------------------
# Hodgkin-Huxley
# ---------------------------------------------------------------------------

class HodgkinHuxley(NeuralModel):
    """
    Conductance-based Hodgkin-Huxley spiking neuron.

    State: [V, m, h, n], shape (N, 4)

    C * dV/dt = I_ext - g_Na*m^3*h*(V-E_Na) - g_K*n^4*(V-E_K) - g_L*(V-E_L)
               + c * W @ V
    dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m
    dh/dt = alpha_h(V)*(1-h) - beta_h(V)*h
    dn/dt = alpha_n(V)*(1-n) - beta_n(V)*n
    """

    state_dim = 4

    def __init__(self, C=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.387,
                 I_ext=10.0, c=0.01):
        self.C = C
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.I_ext = I_ext
        self.c = c

    def _alpha_m(self, V):
        xp = config.xp
        dv = V + 40.0
        # Avoid division by zero
        eps = xp.where(xp.abs(dv) < 1e-7, xp.full_like(dv, 1e-7), dv)
        return 0.1 * eps / (1.0 - xp.exp(-eps / 10.0))

    def _beta_m(self, V):
        xp = config.xp
        return 4.0 * xp.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V):
        xp = config.xp
        return 0.07 * xp.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V):
        xp = config.xp
        return 1.0 / (1.0 + xp.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V):
        xp = config.xp
        dv = V + 55.0
        eps = xp.where(xp.abs(dv) < 1e-7, xp.full_like(dv, 1e-7), dv)
        return 0.01 * eps / (1.0 - xp.exp(-eps / 10.0))

    def _beta_n(self, V):
        xp = config.xp
        return 0.125 * xp.exp(-(V + 65.0) / 80.0)

    def dfdt(self, state, t, W):
        xp = config.xp
        V = state[:, 0]
        m = state[:, 1]
        h = state[:, 2]
        n = state[:, 3]

        I_Na = self.g_Na * m ** 3 * h * (V - self.E_Na)
        I_K  = self.g_K  * n ** 4     * (V - self.E_K)
        I_L  = self.g_L               * (V - self.E_L)

        coupled = self.c * xp.asarray(W @ V).ravel()

        dV = (self.I_ext - I_Na - I_K - I_L + coupled) / self.C
        dm = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dh = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        dn = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n

        return xp.stack([dV, dm, dh, dn], axis=1)

    def initial_state(self, N):
        xp = config.xp
        # Approximate resting state
        V0 = xp.full((N,), -65.0, dtype=xp.float32)
        m0 = xp.full((N,),  0.05, dtype=xp.float32)
        h0 = xp.full((N,),  0.60, dtype=xp.float32)
        n0 = xp.full((N,),  0.32, dtype=xp.float32)
        return xp.stack([V0, m0, h0, n0], axis=1)


# ---------------------------------------------------------------------------
# Kuramoto
# ---------------------------------------------------------------------------

class Kuramoto(NeuralModel):
    """
    Kuramoto phase oscillator model.

    State: [theta], shape (N, 1)  (phase in radians)

    d(theta_i)/dt = omega_i + (K/N) * sum_j W_ij * sin(theta_j - theta_i)

    Parameters
    ----------
    omega : float or array-like of shape (N,)
        Natural frequencies (rad/time). If scalar, all nodes share the same
        frequency.
    K : float
        Global coupling strength.
    """

    state_dim = 1

    def __init__(self, omega=1.0, K=2.0):
        self.omega = omega
        self.K = K

    def dfdt(self, state, t, W):
        xp = config.xp
        theta = state[:, 0]
        N = theta.shape[0]

        # Sparse-friendly decomposition (avoids NxN dense diff matrix):
        #   sum_j W_ij * sin(theta_j - theta_i)
        #   = cos(theta_i) * (W @ sin(theta)) - sin(theta_i) * (W @ cos(theta))
        sin_t = xp.sin(theta)
        cos_t = xp.cos(theta)
        Wsin = xp.asarray(W @ sin_t).ravel()
        Wcos = xp.asarray(W @ cos_t).ravel()
        coupling = (self.K / N) * (cos_t * Wsin - sin_t * Wcos)

        omega = xp.array(self.omega, dtype=xp.float32) if not hasattr(self.omega, "__len__") \
                else xp.array(self.omega, dtype=xp.float32)

        dtheta = omega + coupling
        return xp.expand_dims(dtheta, axis=1)

    def initial_state(self, N):
        xp = config.xp
        rng = np.random.default_rng(0)
        theta0 = xp.array(rng.uniform(0.0, 2.0 * np.pi, size=N).astype(np.float32))
        return xp.expand_dims(theta0, axis=1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS = {
    "WilsonCowan":      WilsonCowan,
    "JansenRit":        JansenRit,
    "IntegrateAndFire": IntegrateAndFire,
    "HodgkinHuxley":    HodgkinHuxley,
    "Kuramoto":         Kuramoto,
}
