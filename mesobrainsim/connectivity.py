"""
Connectivity module: builds the structural adjacency matrix from HDF5 data.

The dataset stores cell-level connectivity in CSR format:
  - 'offset'  : shape (N_cells + 1,) -- row pointers
  - 'indices' : shape (n_edges,)      -- column (target cell) indices

No weight values are stored; all present connections are initialized to 1.

For small N (< DENSE_THRESHOLD), a dense xp array is returned so GPU kernels
can operate on it directly.  For larger N, a scipy.sparse.csr_matrix (CPU) or
cupyx.scipy.sparse.csr_matrix (GPU) is returned.  Both support the @ operator
used by the solvers.
"""

import h5py
import numpy as np
import scipy.sparse as sp
from . import config

# Below this node count use a dense matrix; above use sparse.
DENSE_THRESHOLD = 2000


class Connectivity:
    """
    Builds a weight matrix W for the subsampled node set.

    Attributes
    ----------
    W : dense xp.ndarray or scipy.sparse.csr_matrix, shape (N, N)
        Binary weight matrix (1 where a connection exists, 0 elsewhere).
        Diagonal is 0 (no self-connections).
    D : None
        Distance data is not present in the current dataset.
    is_sparse : bool
        True when W is stored as a sparse matrix.
    """

    def __init__(self, h5_path: str, anatomy):
        self.h5_path = h5_path
        self.indices = anatomy.indices
        self.D = None
        self._load()

    def _load(self):
        xp = config.xp
        node_idx = self.indices   # shape (N,)
        N = len(node_idx)

        with h5py.File(self.h5_path, "r") as f:
            print(f"[Connectivity] HDF5 keys: {list(f.keys())}")

            if "offset" in f and "indices" in f:
                W_sparse = self._build_from_csr(f, node_idx, N)
            else:
                W_sparse = self._try_dense_fallback(f, node_idx, N)

        # Remove self-connections
        W_sparse.setdiag(0)
        W_sparse.eliminate_zeros()

        n_edges = W_sparse.nnz
        print(f"[Connectivity] W shape: ({N}, {N}), edges in subgraph: {n_edges}, sparse: {N >= DENSE_THRESHOLD}")

        if N < DENSE_THRESHOLD:
            self.W = xp.array(W_sparse.toarray(), dtype=xp.float32)
            self.is_sparse = False
        else:
            if config.USE_GPU:
                import cupyx.scipy.sparse as csp
                self.W = csp.csr_matrix(W_sparse, dtype=np.float32)
            else:
                self.W = W_sparse.astype(np.float32)
            self.is_sparse = True

    def _build_from_csr(self, f, node_idx, N):
        """
        Slice the global CSR arrays to build an N×N sparse submatrix.
        Reads only the row slices for selected nodes.
        """
        offset = np.array(f["offset"])       # (N_cells + 1,)
        col_all = np.array(f["indices"])     # (n_edges,)

        # Map global cell index -> local row/col position
        global_to_local = np.full(int(offset.shape[0]) - 1, -1, dtype=np.int32)
        global_to_local[node_idx] = np.arange(N, dtype=np.int32)

        rows, cols = [], []
        for local_i, global_i in enumerate(node_idx):
            start = int(offset[global_i])
            end   = int(offset[global_i + 1])
            targets = col_all[start:end]
            local_targets = global_to_local[targets]
            mask = local_targets >= 0
            local_targets = local_targets[mask]
            if len(local_targets):
                rows.append(np.full(len(local_targets), local_i, dtype=np.int32))
                cols.append(local_targets)

        if rows:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            data = np.ones(len(rows), dtype=np.float32)
        else:
            rows = cols = data = np.array([], dtype=np.float32)

        W_sparse = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
        print(f"[Connectivity] Built submatrix from CSR (no weights -- initialized to 1)")
        return W_sparse

    def _try_dense_fallback(self, f, node_idx, N):
        for key in ["weights", "weight", "W", "connectivity", "adj", "adjacency"]:
            if key in f:
                mat = np.array(f[key])
                if mat.ndim == 2:
                    sub = mat[np.ix_(node_idx, node_idx)].astype(np.float32)
                    print(f"[Connectivity] Loaded dense matrix '{key}', subsampled to ({N}, {N})")
                    return sp.csr_matrix(sub)
        W_np = np.ones((N, N), dtype=np.float32)
        print(f"[Connectivity] No connectivity data found -- initialized W to ones ({N}, {N})")
        return sp.csr_matrix(W_np)

    def load_allen_weights(self, allen_h5_path: str, anatomy):
        """
        Replace binary W with real Allen projection densities.
        Reads /projection_matrix/matrix and /projection_matrix/structure_acronyms.
        Maps structure acronyms -> local node groups via anatomy.region_names.
        Assigns mean projection density as W values for matched node pairs.
        """
        import h5py
        from .utils import resolve_nodes

        with h5py.File(allen_h5_path, 'r') as f:
            matrix = np.array(f['projection_matrix/matrix'])          # (n_exp, n_structs)
            acronyms = [
                a.decode() if isinstance(a, bytes) else str(a)
                for a in f['projection_matrix/structure_acronyms']
            ]

        N = self.W.shape[0]
        # build dense update buffer on CPU
        W_cpu = np.zeros((N, N), dtype=np.float32)

        # mean projection per (source_struct, target_struct) pair
        mean_proj = matrix.mean(axis=0)  # shape (n_structs,) if using col means
        # For pairwise: use outer mean (approx) or iterate experiment injections
        # Full pairwise: for each experiment, injection site = source
        # Here we use the simpler mean_density approach:
        # W[src_nodes, tgt_nodes] = mean projection density from src to tgt
        for i, src_acr in enumerate(acronyms):
            src_idx = resolve_nodes(anatomy, src_acr)
            if not len(src_idx):
                continue
            for j, tgt_acr in enumerate(acronyms):
                tgt_idx = resolve_nodes(anatomy, tgt_acr)
                if not len(tgt_idx):
                    continue
                density = float(matrix[:, j].mean())
                if density > 0:
                    W_cpu[np.ix_(src_idx, tgt_idx)] = density

        np.fill_diagonal(W_cpu, 0)
        xp = config.xp
        if N < DENSE_THRESHOLD:
            self.W = xp.array(W_cpu, dtype=xp.float32)
        else:
            W_sparse = sp.csr_matrix(W_cpu)
            W_sparse.eliminate_zeros()
            if config.USE_GPU:
                import cupyx.scipy.sparse as csp
                self.W = csp.csr_matrix(W_sparse)
            else:
                self.W = W_sparse
        print(f"[Connectivity] Allen weights loaded from '{allen_h5_path}'")

    def __repr__(self):
        n = self.W.shape[0]
        return f"Connectivity(n_nodes={n}, sparse={self.is_sparse}, has_distances={self.D is not None})"
