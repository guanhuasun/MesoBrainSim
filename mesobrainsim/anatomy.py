"""
Anatomy module: loads spatial coordinates and region metadata from an HDF5 file.
"""

import h5py
import numpy as np


class Anatomy:
    """
    Loads and optionally subsamples node positions and region metadata
    from an Allen Brain Cell Atlas HDF5 file.

    Attributes
    ----------
    coords : np.ndarray, shape (N, 3)
        Spatial coordinates [X, Y, Z] for each node.
    region_ids : np.ndarray, shape (N,)
        Numeric region ID for each node.
    region_names : list of str, length N
        Brain area name for each node.
    indices : np.ndarray, shape (N,)
        Original row indices selected from the full dataset (used for
        consistent subsampling in Connectivity).
    """

    def __init__(self, h5_path: str, n_nodes: int = None, subsample_fraction: float = None, seed: int = 0):
        """
        Parameters
        ----------
        h5_path : str
            Path to the HDF5 data file.
        n_nodes : int, optional
            Number of nodes to subsample. Takes priority over subsample_fraction.
        subsample_fraction : float, optional
            Fraction of total nodes to subsample (0, 1].
        seed : int
            Random seed for reproducible subsampling.
        """
        self.h5_path = h5_path
        self._load(n_nodes, subsample_fraction, seed)

    def _load(self, n_nodes, subsample_fraction, seed):
        with h5py.File(self.h5_path, "r") as f:
            self._inspect_keys(f)
            coords_full = self._read_coords(f)
            region_ids_full = self._read_region_ids(f)
            region_names_full = self._read_region_names(f)

        total = len(coords_full)

        if n_nodes is not None:
            n = min(n_nodes, total)
        elif subsample_fraction is not None:
            n = max(1, int(total * subsample_fraction))
        else:
            n = total

        rng = np.random.default_rng(seed)
        self.indices = np.sort(rng.choice(total, size=n, replace=False)) if n < total else np.arange(total)

        self.coords = coords_full[self.indices]
        self.region_ids = region_ids_full[self.indices]
        self.region_names = [region_names_full[i] for i in self.indices]

        print(f"[Anatomy] Loaded {len(self.indices)} / {total} nodes from '{self.h5_path}'")

    def _inspect_keys(self, f):
        """Print top-level HDF5 keys for discovery."""
        print(f"[Anatomy] HDF5 top-level keys: {list(f.keys())}")

    def _read_coords(self, f):
        # Try common key conventions
        for key in ["coords", "coordinates", "xyz", "positions", "cell_locations"]:
            if key in f:
                return np.array(f[key])
        # Try x/y/z as separate datasets
        if all(k in f for k in ("x", "y", "z")):
            x = np.array(f["x"]).ravel()
            y = np.array(f["y"]).ravel()
            z = np.array(f["z"]).ravel()
            return np.stack([x, y, z], axis=1)
        raise KeyError(f"Cannot find coordinate data. Available keys: {list(f.keys())}")

    def _read_region_ids(self, f):
        for key in ["region_id", "region_ids", "structure_id", "parcellation_index", "brain_region_id"]:
            if key in f:
                return np.array(f[key]).ravel()
        return np.zeros(self._n_total(f), dtype=int)

    def _read_region_names(self, f):
        for key in ["region_name", "region_names", "brain_area", "brain_areas", "structure_name"]:
            if key in f:
                raw = f[key]
                return [r.decode() if isinstance(r, bytes) else str(r) for r in raw]
        # Fall back to region_id as string labels
        for key in ["region_id", "region_ids", "structure_id"]:
            if key in f:
                ids = np.array(f[key]).ravel()
                return [f"region_{rid}" for rid in ids]
        n = self._read_coords(f).shape[0]
        return [f"region_{i}" for i in range(n)]

    def _n_total(self, f):
        return self._read_coords(f).shape[0]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"Anatomy(n_nodes={len(self)}, h5_path='{self.h5_path}')"
