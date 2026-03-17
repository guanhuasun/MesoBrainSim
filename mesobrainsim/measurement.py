"""Record simulation state to HDF5. Preallocates datasets for zero-copy streaming."""

import numpy as np
import h5py
from . import config
from .utils import resolve_nodes


class Probe:
    def __init__(self, name, node_indices, var_index=0, use_neighborhood_avg=False):
        self.name = name
        self.node_indices = np.asarray(node_indices, dtype=np.intp)
        self.var_index = int(var_index)
        self.use_neighborhood_avg = use_neighborhood_avg


class MeasurementHook:
    """
    Streams state snapshots to an HDF5 file.

    HDF5 schema
    -----------
    /metadata/coords       (k, 3)
    /metadata/region_ids   (k,)
    /metadata/times        (T_rec,)
    /probes/<name>/data    (T_rec, k)
    /probes/<name>/node_ids (k,)
    /probes/<name>/var_index scalar
    """

    def __init__(self, probes, h5_path, anatomy, W, n_recorded):
        self.probes = probes
        self.h5_path = h5_path
        self.anatomy = anatomy
        self.W = W
        self.n_recorded = n_recorded
        self._file = None
        self._ds_times = None
        self._ds_data = {}
        self._cursor = 0

    def open(self):
        self._cursor = 0
        f = h5py.File(self.h5_path, 'w')
        self._file = f

        f.create_dataset('metadata/times', shape=(self.n_recorded,), dtype='float32')
        self._ds_times = f['metadata/times']

        # global metadata (all N nodes)
        f.create_dataset('metadata/coords',
                         data=self.anatomy.coords.astype('float32'),
                         compression='gzip')
        f.create_dataset('metadata/region_ids',
                         data=self.anatomy.region_ids.astype('int32'))

        for probe in self.probes:
            idx = probe.node_indices
            k = len(idx)
            grp = f.require_group(f'probes/{probe.name}')
            grp.create_dataset('node_ids', data=idx.astype('int32'))
            grp.attrs['var_index'] = probe.var_index
            grp.create_dataset('data', shape=(self.n_recorded, k), dtype='float32')
            self._ds_data[probe.name] = grp['data']

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __call__(self, step, t, state):
        if self._cursor >= self.n_recorded:
            return
        self._ds_times[self._cursor] = t
        for probe in self.probes:
            if probe.use_neighborhood_avg:
                sig = self._neighborhood_avg(state[:, probe.var_index], self.W)
                vals = sig[probe.node_indices]
            else:
                vals = state[probe.node_indices, probe.var_index]
            # transfer to CPU if on GPU
            if config.USE_GPU and hasattr(vals, 'get'):
                vals = vals.get()
            self._ds_data[probe.name][self._cursor] = np.asarray(vals, dtype='float32')
        self._cursor += 1

    def _neighborhood_avg(self, signal, W):
        """(W @ signal) for each node; returns N-vector."""
        xp = config.xp
        N = signal.shape[0]
        is_dense_small_gpu = config.USE_GPU and not hasattr(W, 'indptr') and N < 2000
        if is_dense_small_gpu:
            try:
                import cupy as cp
                return cp.dot(W, signal)
            except Exception:
                pass
        return xp.asarray(W @ signal).ravel()
