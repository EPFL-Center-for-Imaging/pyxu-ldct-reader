import numpy as np


def _get_src_pos(dataset):
    radial0 = dataset.DetectorFocalCenterRadialDistance
    angular0 = dataset.DetectorFocalCenterAngularPosition
    axial0 = dataset.DetectorFocalCenterAxialPosition

    radial_delta = dataset.SourceRadialDistanceShift
    angular_delta = dataset.SourceAngularPositionShift
    axial_delta = dataset.SourceAxialPositionShift

    return cylindrical_to_cartesian(
        radial=radial0 + radial_delta, angular=angular0 + angular_delta, axial=axial0 + axial_delta
    )


def _get_det_pos(dataset):
    radial0 = dataset.DetectorFocalCenterRadialDistance
    angular0 = dataset.DetectorFocalCenterAngularPosition
    axial0 = dataset.DetectorFocalCenterAxialPosition
    d0 = dataset.ConstantRadialDistance

    width_col = dataset.DetectorElementTransverseSpacing
    width_row = dataset.DetectorElementAxialSpacing
    central_col, central_row = dataset.DetectorCentralElement
    ncolumns = dataset.NumberofDetectorColumns
    nrows = dataset.NumberofDetectorRows

    if dataset.DetectorShape == "CYLINDRICAL":
        radial = d0 - radial0
        column_arc_distance = ((np.arange(ncolumns, dtype="float32") + 1) - central_col) * width_col
        row_distance = (central_row - (np.arange(nrows, dtype="float32") + 1)) * width_row
        theta = angular0 + np.pi + column_arc_distance / d0
        theta = theta % (2 * np.pi)  # wrap between 0, 2pi
        theta = np.tile(theta.reshape(ncolumns, 1), (1, nrows))

        axial_ij = axial0 + row_distance
        axial_ij = np.tile(axial_ij.reshape(1, nrows), (ncolumns, 1))

    else:
        raise NotImplementedError

    return cylindrical_to_cartesian(radial=radial, angular=theta, axial=axial_ij)


def cylindrical_to_cartesian(radial, angular, axial):
    x = radial * np.cos(-angular)
    y = radial * np.sin(-angular)
    z = -axial
    return x, y, z
