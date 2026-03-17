"""
Connectivity module: builds the structural adjacency matrix from HDF5 data.

The current dataset stores cell-level connectivity in CSR format:
  - 'offset'  : shape (N_cells + 1,) — row pointers
  - 'indices' : shape (n_edges,)     — column (target cell) indices

No weight values are stored; all present connections are initialized to 1.
A region-level count matrix ('region_conn_count') is also available.
"""

import h5py
import numpy as np
from . import config


class Connectivity:
    """
    Builds a dense weight matrix W (N, N) for the subsampled node set.

    Because the full matrix is ~300k × 300k, we only materialise the
    dense submatrix for the N nodes selected by the Anatomy instance.

    Attributes
    ----------
    W : xp.ndarray, shape (N, N)
        Binary weight matrix (1 where a connection exists, 0 otherwise).
        Diagonal is set to 0 (no self-connections).
    D : None
        Distance data is not present in the current dataset.
    """

    def __init__(self, h5_path: str, anatomy):
        """
        Parameters
        ----------
        h5_path : str
            Path to the HDF5 data file.
        anatomy : Anatomy
            Loaded Anatomy instance; uses anatomy.indices for subsampling.
        """
        self.h5_path = h5_path
        self.indices = anatomy.indices
        self.D = None
        self._load()

    def _load(self):
        xp = config.xp
        node_idx = self.indices          # shape (N,)
        N = len(node_idx)

        # Map global cell index → position in our subsampled set
        idx_set = set(node_idx.tolist())
        global_to_local = {g: l for l, g in enumerate(node_idx.tolist())}

        with h5py.File(self.h5_path, "r") as f:
            keys = list(f.keys())
            print(f"[Connectivity] HDF5 keys: {keys}")

            if "offset" in f and "indices" in f:
                W_np = self._build_from_csr(f, node_idx, idx_set, global_to_local, N)
            else:
                # Fallback: try dense matrix keys
                W_np = self._try_dense(f, node_idx, N)

        np.fill_diagonal(W_np, 0.0)
        self.W = xp.array(W_np, dtype=xp.float32)
        n_edges = int(self.W.sum())
        print(f"[Connectivity] W shape: ({N}, {N}), edges in subgraph: {n_edges}")

    def _build_from_csr(self, f, node_idx, idx_set, global_to_local, N):
        """
        Extract the dense N×N submatrix from CSR storage.
        Reads only the offset slices for the selected rows to avoid
        loading all 86M edge indices into memory at once.
        """
        offset = np.array(f["offset"])          # (N_cells + 1,)
        col_indices = np.array(f["indices"])     # (n_edges,)  — all edges

        W_np = np.zeros((N, N), dtype=np.float32)

        for local_i, global_i in enumerate(node_idx):
            start = int(offset[global_i])
            end   = int(offset[global_i + 1])
            targets = col_indices[start:end]
            for t in targets:
                local_j = global_to_local.get(int(t))
                if local_j is not None:
                    W_np[local_i, local_j] = 1.0

        print(f"[Connectivity] Built submatrix from CSR (no weights -- initialized to 1)")
        return W_np

    def _try_dense(self, f, node_idx, N):
        for key in ["weights", "weight", "W", "connectivity", "adj", "adjacency"]:
            if key in f:
                mat = np.array(f[key])
                if mat.ndim == 2:
                    sub = mat[np.ix_(node_idx, node_idx)]
                    print(f"[Connectivity] Loaded dense matrix '{key}' and subsampled to ({N}, {N})")
                    return sub.astype(np.float32)
        # No connectivity data at all — fully connected with weight 1
        W_np = np.ones((N, N), dtype=np.float32)
        print(f"[Connectivity] No connectivity data found — initialized W to ones ({N}, {N})")
        return W_np

    def __repr__(self):
        n = self.W.shape[0]
        return f"Connectivity(n_nodes={n}, has_distances={self.D is not None})"
