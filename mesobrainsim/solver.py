"""
Numerical solvers: Euler and Heun (2nd-order Runge-Kutta).
Both are fully vectorized over N nodes using the configured xp backend.
"""

from . import config


def _is_sparse(W):
    return hasattr(W, 'indptr')


class EulerSolver:
    def run(self, model, W, T: float, dt: float, record_every: int = 1,
            hooks=None, stimulator=None, plasticity_hooks=None,
            return_trajectory=True):
        xp = config.xp
        hooks = hooks or []
        plasticity_hooks = plasticity_hooks or []
        N = W.shape[0]
        n_steps = int(T / dt)
        is_sp = _is_sparse(W)

        for ph in plasticity_hooks:
            ph.attach(W, is_sp)
        if stimulator is not None:
            stimulator.prepare(N)
            effective_dfdt = stimulator.get_shim(model)
        else:
            effective_dfdt = model.dfdt
        for h in hooks:
            h.open()

        state = model.initial_state(N)
        trajectory, times = [], []

        for step in range(n_steps):
            t = step * dt
            if stimulator is not None:
                stimulator(t)
            if step % record_every == 0:
                if return_trajectory:
                    trajectory.append(state)
                    times.append(t)
                for h in hooks:
                    h(step, t, state)
            state = state + dt * effective_dfdt(state, t, W)
            for ph in plasticity_hooks:
                ph(step, t, state)

        # final frame
        t_final = n_steps * dt
        if return_trajectory:
            trajectory.append(state)
            times.append(t_final)
        for h in hooks:
            h(n_steps, t_final, state)
        for h in hooks:
            h.close()

        if return_trajectory:
            return xp.stack(trajectory, axis=0), xp.array(times, dtype=xp.float32)
        return None, None


class HeunSolver:
    def run(self, model, W, T: float, dt: float, record_every: int = 1,
            hooks=None, stimulator=None, plasticity_hooks=None,
            return_trajectory=True):
        xp = config.xp
        hooks = hooks or []
        plasticity_hooks = plasticity_hooks or []
        N = W.shape[0]
        n_steps = int(T / dt)
        is_sp = _is_sparse(W)

        for ph in plasticity_hooks:
            ph.attach(W, is_sp)
        if stimulator is not None:
            stimulator.prepare(N)
            effective_dfdt = stimulator.get_shim(model)
        else:
            effective_dfdt = model.dfdt
        for h in hooks:
            h.open()

        state = model.initial_state(N)
        trajectory, times = [], []

        for step in range(n_steps):
            t = step * dt
            if stimulator is not None:
                stimulator(t)
            if step % record_every == 0:
                if return_trajectory:
                    trajectory.append(state)
                    times.append(t)
                for h in hooks:
                    h(step, t, state)
            k1 = effective_dfdt(state, t, W)
            k2 = effective_dfdt(state + dt * k1, t + dt, W)
            state = state + (dt / 2.0) * (k1 + k2)
            for ph in plasticity_hooks:
                ph(step, t, state)

        # final frame
        t_final = n_steps * dt
        if return_trajectory:
            trajectory.append(state)
            times.append(t_final)
        for h in hooks:
            h(n_steps, t_final, state)
        for h in hooks:
            h.close()

        if return_trajectory:
            return xp.stack(trajectory, axis=0), xp.array(times, dtype=xp.float32)
        return None, None


SOLVERS = {
    "Euler": EulerSolver,
    "Heun":  HeunSolver,
}
