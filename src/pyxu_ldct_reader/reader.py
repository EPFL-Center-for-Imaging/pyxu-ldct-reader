import os

import numpy as np

from pyxu_ldct_reader.dicom_utils import _read_dicom_file
from pyxu_ldct_reader.geometry_utils import _get_det_pos, _get_src_pos


def load_projections(folder, indices=None):
    """
    Load geometry and data stored in Mayo format from folder and prepare it for
    XRayTransform.

    Parameters
    ----------
    folder : str
        Path to the folder where the LDCT-projection DICOM files are stored.
    indices : optional
        Indices of the projections to load.
        Accepts advanced indexing such as slice or list of indices.

    Returns
    -------
    proj_data : `numpy.ndarray`
        Projection data, given as the line integral of the linear attenuation
        coefficient (g/cm^3). Its unit is thus g/cm^2.
    n_spec : `numpy.ndarray`
        Ray directions for each projection.
    t_spec : `numpy.ndarray`
        Offset specifications for each projection.
    """

    if not os.path.isdir(folder):
        raise ValueError(f"Invalid folder path: {folder}")

    file_names = sorted(f for f in os.listdir(folder) if f.endswith(".dcm"))
    if not file_names:
        raise ValueError(f"No DICOM files found in {folder}")

    if indices is not None:
        file_names = [file_names[i] for i in indices]

    data_array, n_spec, t_spec = [], [], []
    for i, file_name in enumerate(file_names):
        dataset = _read_dicom_file(os.path.join(folder, file_name), file_name)
        if i == 0:
            dataset0 = dataset
            proj_array, n_spec_val, t_spec_val = _process_projection_data(dataset, i, None)
        else:
            proj_array, n_spec_val, t_spec_val = _process_projection_data(dataset, i, dataset0)
        data_array.append(proj_array)
        n_spec.append(n_spec_val)
        t_spec.append(t_spec_val)

    return np.array(data_array), np.array(n_spec), np.array(t_spec)


def _process_projection_data(dataset, index, dataset0):
    # A bit confusing
    rows = dataset.Rows  # NumberofDetectorColumns
    cols = dataset.Columns  # NumberofDetectorRows
    rescale_intercept = getattr(dataset, "RescaleIntercept", 0)
    rescale_slope = getattr(dataset, "RescaleSlope", 1)

    if index == 0:
        _check_dataset_consistency(dataset, rows, cols, rescale_intercept, rescale_slope)
    elif dataset0:
        _check_dataset_consistency(dataset0, rows, cols, rescale_intercept, rescale_slope)

    proj_array = np.frombuffer(dataset.PixelData, dtype="H").reshape(rows, cols).astype("float32")
    proj_array *= rescale_slope
    proj_array += rescale_intercept

    # Get source and detector cartesian coordinates
    src_pos = np.stack(_get_src_pos(dataset), dtype="float32")[np.newaxis, np.newaxis]
    det_pos = np.dstack(_get_det_pos(dataset))

    # Compute the ray direction (unit vector)
    n_spec_val = det_pos - src_pos
    n_spec_val /= np.linalg.norm(n_spec_val, axis=-1, keepdims=True)

    # Compute the offset (t_spec)
    t_spec_val = np.tile(src_pos, (*det_pos.shape[:-1], 1))

    return proj_array, n_spec_val, t_spec_val


def _check_dataset_consistency(reference_dataset, rows, cols, rescale_intercept, rescale_slope):
    assert rows == reference_dataset.Rows
    assert cols == reference_dataset.Columns
    assert rescale_intercept == getattr(reference_dataset, "RescaleIntercept", 0)
    assert rescale_slope == getattr(reference_dataset, "RescaleSlope", 1)
