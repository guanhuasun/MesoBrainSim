"""
Numerical solvers: Euler and Heun (2nd-order Runge-Kutta) methods.
Both are fully vectorized over N nodes using the configured xp backend.
"""

from . import config


class EulerSolver:
    """
    Explicit Euler integrator.

    state_{n+1} = state_n + dt * f(state_n, t_n, W)
    """

    def run(self, model, W, T: float, dt: float, record_every: int = 1):
        """
        Integrate the model forward in time.

        Parameters
        ----------
        model : NeuralModel
            Instantiated neural model.
        W : xp.ndarray, shape (N, N)
            Weight matrix.
        T : float
            Total simulation time.
        dt : float
            Time step.
        record_every : int
            Store state every this many steps (reduces memory for long sims).

        Returns
        -------
        trajectory : xp.ndarray, shape (n_recorded, N, state_dim)
        times : xp.ndarray, shape (n_recorded,)
        """
        xp = config.xp
        N = W.shape[0]
        n_steps = int(T / dt)

        state = model.initial_state(N)
        trajectory, times = [], []

        for step in range(n_steps):
            t = step * dt
            if step % record_every == 0:
                trajectory.append(state)
                times.append(t)
            state = state + dt * model.dfdt(state, t, W)

        trajectory.append(state)
        times.append(n_steps * dt)

        return xp.stack(trajectory, axis=0), xp.array(times, dtype=xp.float32)


class HeunSolver:
    """
    Heun's method (explicit trapezoidal rule, 2nd-order Runge-Kutta).

    k1 = f(state_n, t_n, W)
    k2 = f(state_n + dt*k1, t_n + dt, W)
    state_{n+1} = state_n + (dt/2) * (k1 + k2)
    """

    def run(self, model, W, T: float, dt: float, record_every: int = 1):
        """
        Same signature as EulerSolver.run.
        """
        xp = config.xp
        N = W.shape[0]
        n_steps = int(T / dt)

        state = model.initial_state(N)
        trajectory, times = [], []

        for step in range(n_steps):
            t = step * dt
            if step % record_every == 0:
                trajectory.append(state)
                times.append(t)

            k1 = model.dfdt(state, t, W)
            k2 = model.dfdt(state + dt * k1, t + dt, W)
            state = state + (dt / 2.0) * (k1 + k2)

        trajectory.append(state)
        times.append(n_steps * dt)

        return xp.stack(trajectory, axis=0), xp.array(times, dtype=xp.float32)


SOLVERS = {
    "Euler": EulerSolver,
    "Heun":  HeunSolver,
}
