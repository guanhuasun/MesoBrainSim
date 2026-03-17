"""Shared utilities; imported by stimulation and measurement to avoid circular imports."""

import numpy as np


def resolve_nodes(anatomy, selector):
    """
    Return local indices 0..N-1 matching selector.

    selector:
      int          -> match region_id
      str          -> exact region_name match, then substring, then int cast
      list[int]    -> pass through as np.intp array
      np.ndarray   -> pass through as np.intp array
    """
    if isinstance(selector, np.ndarray):
        return selector.astype(np.intp)
    if isinstance(selector, (list, tuple)):
        arr = np.asarray(selector)
        if arr.dtype.kind in ('i', 'u'):
            return arr.astype(np.intp)
        # list of strings: recurse
        idx = np.concatenate([resolve_nodes(anatomy, s) for s in selector]) if len(selector) else np.array([], dtype=np.intp)
        return np.unique(idx).astype(np.intp)
    if isinstance(selector, (int, np.integer)):
        return np.where(anatomy.region_ids == int(selector))[0].astype(np.intp)
    if isinstance(selector, str):
        names = anatomy.region_names
        # exact match
        exact = np.array([i for i, n in enumerate(names) if n == selector], dtype=np.intp)
        if len(exact):
            return exact
        # substring
        sub = np.array([i for i, n in enumerate(names) if selector in n], dtype=np.intp)
        if len(sub):
            return sub
        # try region_id numeric
        try:
            rid = int(selector)
            return np.where(anatomy.region_ids == rid)[0].astype(np.intp)
        except ValueError:
            pass
        return np.array([], dtype=np.intp)
    return np.asarray(selector, dtype=np.intp)
