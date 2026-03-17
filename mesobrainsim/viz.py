"""
Visualization module: 3D point cloud and animated signal propagation via PyVista.
"""

import numpy as np


class BrainVisualizer:
    """
    Wraps PyVista to render simulation activity on 3D node coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Spatial coordinates [X, Y, Z] for each node.
    trajectory : np.ndarray, shape (T, N, state_dim)
        Simulation output. The first state variable (index 0) is used as
        the scalar for coloring.
    times : np.ndarray, shape (T,)
        Time axis.
    """

    def __init__(self, coords, trajectory, times):
        self.coords = np.array(coords, dtype=np.float32)
        # Convert from xp array to numpy if needed
        self.trajectory = self._to_numpy(trajectory)
        self.times = self._to_numpy(times)

    @staticmethod
    def _to_numpy(arr):
        if hasattr(arr, "get"):        # CuPy array
            return arr.get()
        return np.asarray(arr)

    def _build_cloud(self, activity):
        import pyvista as pv
        cloud = pv.PolyData(self.coords)
        cloud["activity"] = activity.astype(np.float32)
        return cloud

    def plot_static(self, t_index: int = -1, cmap: str = "hot",
                    point_size: float = 5.0, show: bool = True):
        """
        Render a static point cloud colored by activity at one time step.

        Parameters
        ----------
        t_index : int
            Index into the time axis. Defaults to the last frame.
        cmap : str
            Matplotlib colormap name.
        point_size : float
            Rendered point size.
        show : bool
            If True, open an interactive window.
        """
        import pyvista as pv
        activity = self.trajectory[t_index, :, 0]
        cloud = self._build_cloud(activity)

        pl = pv.Plotter(title=f"Brain activity  t = {self.times[t_index]:.1f} ms")
        pl.add_points(cloud, scalars="activity", cmap=cmap,
                      point_size=point_size, render_points_as_spheres=True)
        pl.add_scalar_bar(title="Activity")
        pl.show_axes()
        if show:
            pl.show()
        return pl

    def animate(self, output_path: str = "brain_sim.gif",
                cmap: str = "hot", point_size: float = 5.0,
                framerate: int = 15, step: int = 1):
        """
        Render an animated GIF/MP4 of signal propagation over time.

        Parameters
        ----------
        output_path : str
            Output file path (.gif or .mp4).
        cmap : str
            Matplotlib colormap name.
        point_size : float
            Rendered point size.
        framerate : int
            Frames per second.
        step : int
            Use every `step`-th time frame to speed up rendering.
        """
        import pyvista as pv

        frames = range(0, len(self.times), step)
        activity_all = self.trajectory[::step, :, 0]
        vmin = float(activity_all.min())
        vmax = float(activity_all.max())

        pl = pv.Plotter(off_screen=True)
        first_activity = activity_all[0]
        cloud = self._build_cloud(first_activity)
        pl.add_points(cloud, scalars="activity", cmap=cmap,
                      clim=[vmin, vmax], point_size=point_size,
                      render_points_as_spheres=True)
        pl.add_scalar_bar(title="Activity")
        pl.show_axes()

        if output_path.endswith(".gif"):
            pl.open_gif(output_path, fps=framerate)
        elif output_path.endswith(".mp4"):
            pl.open_movie(output_path, framerate=framerate)
        else:
            raise ValueError("output_path must end in .gif or .mp4")

        for i, activity in enumerate(activity_all):
            cloud["activity"] = activity.astype(np.float32)
            pl.write_frame()

        pl.close()
        print(f"[Visualizer] Animation saved to '{output_path}'")
