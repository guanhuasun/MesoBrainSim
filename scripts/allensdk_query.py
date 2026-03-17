"""
Allen Brain Connectivity Atlas query.
Run in 'allen' conda env:
  conda run -n allen python scripts/allensdk_query.py --regions VISp,MOs --output data/allen_connectivity.h5

HDF5 output schema:
  /structures/ids, acronyms, names, rgb, paths
  /projection_matrix/matrix (n_exp, n_structs), experiment_ids, structure_acronyms
  /experiments/ids, structure_ids, injection_volumes, injection_x, injection_y, injection_z
"""

import argparse
import numpy as np
import h5py
import pandas as pd


def query_region_metadata(mcc, tree, acronyms, ids):
    structs = []
    for acr in acronyms:
        s = tree.get_structures_by_acronym([acr])
        if s:
            structs.extend(s)
    for sid in ids:
        s = tree.get_structures_by_id([sid])
        if s:
            structs.extend(s)
    # deduplicate by id
    seen = set()
    out = []
    for s in structs:
        if s['id'] not in seen:
            seen.add(s['id'])
            out.append(s)
    return out


def query_projection_matrix(mcc, tree, structure_ids):
    """
    Returns (matrix, experiment_ids, col_acronyms).
    matrix shape: (n_experiments, n_structures)
    """
    pm, exp_ids = mcc.get_projection_matrix(
        experiment_ids=None,
        projection_structure_ids=structure_ids,
        parameter='projection_density',
    )
    col_acronyms = [tree.get_structures_by_id([sid])[0]['acronym']
                    for sid in structure_ids]
    return pm, exp_ids, col_acronyms


def query_experiments_by_injection(mcc, structure_ids):
    exps = mcc.get_experiments(injection_structure_ids=structure_ids)
    return pd.DataFrame(exps)


def save_to_h5(path, structs, matrix, exp_ids, col_acronyms, exps_df):
    with h5py.File(path, 'w') as f:
        sg = f.create_group('structures')
        sg.create_dataset('ids', data=np.array([s['id'] for s in structs], dtype='int32'))
        sg.create_dataset('acronyms',
                          data=np.array([s['acronym'].encode() for s in structs]))
        sg.create_dataset('names',
                          data=np.array([s['name'].encode() for s in structs]))
        sg.create_dataset('rgb',
                          data=np.array([s.get('rgb_triplet', [0, 0, 0]) for s in structs],
                                        dtype='uint8'))
        sg.create_dataset('paths',
                          data=np.array([s.get('structure_id_path', '').encode()
                                         for s in structs]))

        pg = f.create_group('projection_matrix')
        pg.create_dataset('matrix', data=matrix.astype('float32'), compression='gzip')
        pg.create_dataset('experiment_ids', data=np.array(exp_ids, dtype='int32'))
        pg.create_dataset('structure_acronyms',
                          data=np.array([a.encode() for a in col_acronyms]))

        eg = f.create_group('experiments')
        if len(exps_df):
            eg.create_dataset('ids', data=exps_df['id'].values.astype('int32'))
            eg.create_dataset('structure_ids',
                              data=exps_df['structure_id'].values.astype('int32'))
            eg.create_dataset('injection_volumes',
                              data=exps_df.get('injection_volume',
                                               pd.Series(np.zeros(len(exps_df)))).values.astype('float32'))
            for axis in ('x', 'y', 'z'):
                col = f'injection_{axis}'
                vals = exps_df.get(col, pd.Series(np.zeros(len(exps_df)))).values
                eg.create_dataset(f'injection_{axis}', data=vals.astype('float32'))

    print(f"[AllenSDK] Saved to '{path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions', default='VISp,MOs',
                        help='Comma-separated Allen acronyms, e.g. VISp,MOs')
    parser.add_argument('--output', default='data/allen_connectivity.h5')
    args = parser.parse_args()

    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        raise ImportError("Run in 'allen' conda env: conda run -n allen python ...")

    mcc = MouseConnectivityCache(resolution=25)
    tree = mcc.get_structure_tree()

    acronyms = [a.strip() for a in args.regions.split(',')]
    structure_ids = [
        tree.get_structures_by_acronym([a])[0]['id']
        for a in acronyms
        if tree.get_structures_by_acronym([a])
    ]
    print(f"[AllenSDK] Querying {len(structure_ids)} structures: {acronyms}")

    structs = query_region_metadata(mcc, tree, acronyms, structure_ids)
    matrix, exp_ids, col_acronyms = query_projection_matrix(mcc, tree, structure_ids)
    exps_df = query_experiments_by_injection(mcc, structure_ids)

    print(f"[AllenSDK] matrix shape: {matrix.shape}, experiments: {len(exp_ids)}")
    save_to_h5(args.output, structs, matrix, exp_ids, col_acronyms, exps_df)


if __name__ == '__main__':
    main()
