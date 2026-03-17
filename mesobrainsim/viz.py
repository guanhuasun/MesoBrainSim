import numpy as np


class BrainVisualizer:
    def __init__(self, coords, trajectory, times):
        coords = np.array(coords, dtype=np.float32)
        coords[:, 2] = 11400.0 - coords[:, 2]   # z-flip anatomical orientation
        self.coords     = coords / 100.0          # µm -> 0.1mm (sfn24 scale)
        self.trajectory = self._to_numpy(trajectory)
        self.times      = self._to_numpy(times)

    @staticmethod
    def _to_numpy(arr):
        return arr.get() if hasattr(arr, 'get') else np.asarray(arr)

    @staticmethod
    def _set_camera(pl):
        pl.camera_position = 'xz'
        pl.camera.roll      = -90
        pl.camera.elevation = 15
        pl.camera.azimuth   = 30
        pl.camera.zoom(1.3)
        pl.set_background(color='k')

    def plot_static(self, t_index=-1, cmap='turbo', point_size=3.0,
                    clim=None, show=True, window_size=(600, 800)):
        import pyvista as pv
        activity = self.trajectory[t_index, :, 0].astype(np.float32)
        if clim is None:
            clim = (float(np.nanpercentile(activity, 5)),
                    float(np.nanpercentile(activity, 95)))

        mesh = pv.PolyData(self.coords)
        mesh['scalars'] = activity

        pl = pv.Plotter(off_screen=not show, window_size=list(window_size))
        pl.add_mesh(mesh, scalars='scalars', cmap=cmap,
                    render_points_as_spheres=False, nan_opacity=0.999,
                    point_size=point_size, opacity=1, lighting=True,
                    clim=list(clim), scalar_bar_args=dict(color='w'))
        pl.update_scalar_bar_range(list(clim))
        self._set_camera(pl)
        if show:
            pl.show()
        return pl

    def animate(self, output_path='brain_sim.gif', cmap='turbo',
                point_size=3.0, framerate=15, step=1,
                clim=None, window_size=(600, 800)):
        import pyvista as pv

        activity_all = self.trajectory[::step, :, 0].astype(np.float32)
        times_all    = self.times[::step]
        if clim is None:
            clim = (float(np.nanpercentile(activity_all, 25)),
                    float(np.nanpercentile(activity_all, 95)))

        mesh = pv.PolyData(self.coords)
        mesh['Voltage'] = activity_all[0]

        pl = pv.Plotter(off_screen=True, window_size=list(window_size))
        pl.add_mesh(mesh, scalars='Voltage', cmap=cmap,
                    render_points_as_spheres=False, nan_opacity=0.999,
                    point_size=point_size, opacity=1, lighting=True,
                    clim=list(clim), scalar_bar_args=dict(color='w'))
        pl.update_scalar_bar_range(list(clim))
        self._set_camera(pl)

        if output_path.endswith('.gif'):
            pl.open_gif(output_path, fps=framerate)
        elif output_path.endswith('.mp4'):
            pl.open_movie(output_path, framerate=framerate)
        else:
            raise ValueError('output_path must end in .gif or .mp4')

        for i, t in enumerate(times_all):
            mesh['Voltage'] = activity_all[i]
            pl.add_text(f'Time: {t:.1f} ms', name='time-label',
                        font_size=18, color='w')
            pl.update_scalar_bar_range(list(clim))
            pl.write_frame()
            pl.render()

        pl.close()
        print(f"[Visualizer] Animation saved to '{output_path}'")
