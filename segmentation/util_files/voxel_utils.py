# voxel_utils.py
"""
Utility functions for fetching volumetric data from neuroglancer datasets.

Provides batched access to FlyEM Hemibrain and Optic Lobe datasets via tensorstore.
Handles boundary conditions by zero-padding when requested volumes extend beyond
dataset bounds.

Supported data types:
    - grayscale: Raw EM data
    - grayscale_clahe: Contrast-enhanced EM data (CLAHE)
    - segmentation: Neuron body ID segmentation
    - mito-objects: Mitochondria segmentation
    - rois: Region of interest labels
"""
import numpy as np
import tensorstore as ts


def get_subvols_batched(init_boxes_zyx, datatype, dataset='hemibrain'):
    """
    Fetch multiple subvolumes in a batched manner using tensorstore.

    Args:
        init_boxes_zyx: List of bounding boxes, each as [[min_z, min_y, min_x], [max_z, max_y, max_x]]
        datatype: Type of data to fetch ('grayscale', 'grayscale_clahe', 'segmentation', etc.)
        dataset: Dataset to use ('hemibrain' or 'optic-lobe')

    Returns:
        np.ndarray of subvolumes with shape (n_boxes, z, y, x)
    """
    if dataset == 'hemibrain':
        kvstore_map = {
            'grayscale': 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg',
            'grayscale_clahe': 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg',
            'segmentation': 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/',
            'mito-objects': 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/mito-objects-grouped',
            'rois': 'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/rois'
        }
    elif dataset == 'optic-lobe':
        kvstore_map = {
            'grayscale': 'gs://neuroglancer-janelia-flyem-optic-lobe/emdata/raw/jpeg',
            'grayscale_clahe': 'gs://neuroglancer-janelia-flyem-optic-lobe/emdata/clahe_yz/jpeg',
            'segmentation': 'gs://neuroglancer-janelia-flyem-optic-lobe/v1.0/segmentation/',
            'mito-objects': 'gs://neuroglancer-janelia-flyem-optic-lobe/v1.0/mito-objects-grouped',
            'rois': 'gs://neuroglancer-janelia-flyem-optic-lobe/v1.0/rois'
        }
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Use 'hemibrain' or 'optic-lobe'")

    kvstore = kvstore_map.get(datatype)
    if kvstore is None:
        raise ValueError(f"Unsupported datatype: {datatype}")

    dataset_3d = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': kvstore,
        'context': {'cache_pool': {'total_bytes_limit': 16_000_000_000}},
        'recheck_cached_data': 'open'
    }).result()[ts.d['channel'][0]]

    min_point_zyx = np.flip(np.array([0, 0, 0], dtype=int))
    max_point_zyx = np.flip(np.array([34367, 37888, 41344], dtype=int))

    def process_subvolume(subvol, context):
        box_zyx = context['box_zyx']
        prepend_zyx = context['prepend_zyx']
        append_zyx = context['append_zyx']

        subvol = np.transpose(subvol, axes=[2, 1, 0])  # subvol_zyx

        for i_axis in range(3):
            if prepend_zyx[i_axis] > 0:
                box_zyx[0][i_axis] -= prepend_zyx[i_axis]
                prepend_shape = list(subvol.shape)
                prepend_shape[i_axis] = prepend_zyx[i_axis]
                subvol = np.concatenate([np.zeros(prepend_shape), subvol], axis=i_axis)
            if append_zyx[i_axis] > 0:
                box_zyx[1][i_axis] += append_zyx[i_axis]
                append_shape = list(subvol.shape)
                append_shape[i_axis] = append_zyx[i_axis]
                subvol = np.concatenate([subvol, np.zeros(append_shape)], axis=i_axis)

        return subvol

    with ts.Batch() as batch:
        futures_and_contexts = []
        for init_box_zyx in (np.array(b).astype(int) for b in init_boxes_zyx):
            min_box_zyx, max_box_zyx = init_box_zyx
            box_zyx = init_box_zyx.copy()

            prepend_zyx = np.maximum(min_point_zyx - init_box_zyx[0], 0)
            min_box_zyx = np.maximum(min_box_zyx, min_point_zyx)

            append_zyx = np.maximum(init_box_zyx[1] - max_point_zyx, 0)
            max_box_zyx = np.minimum(max_box_zyx, max_point_zyx)

            min_box_zyx = np.array(min_box_zyx).astype(int)
            max_box_zyx = np.array(max_box_zyx).astype(int)

            future = dataset_3d[
                min_box_zyx[2]:max_box_zyx[2],
                min_box_zyx[1]:max_box_zyx[1],
                min_box_zyx[0]:max_box_zyx[0]
            ].read(batch=batch)

            context = {
                'init_box_zyx': init_box_zyx,
                'box_zyx': box_zyx,
                'prepend_zyx': prepend_zyx,
                'append_zyx': append_zyx
            }
            futures_and_contexts.append((future, context))

    return np.array([process_subvolume(future.result(), context) for future, context in futures_and_contexts])
