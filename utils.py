import numpy as np


from typing import Any


def is_point_in_ROI(point: np.ndarray[Any, Any], center: np.ndarray[Any, Any], size: np.ndarray[Any, Any]) -> bool:
    """
    Check if a 3D point is within a rectangular region of interest (ROI).

    :param point: The point to check (3 floats).
    :param center: The center of the ROI (3 floats).
    :param size: The size of the ROI (3 floats).
    :return: True if the point is within the ROI, False otherwise.
    """
    size_half = size / 2.0
    local_point = point - center

    return bool(np.all(np.abs(local_point) <= size_half))
