import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

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


def did_rad_check_recurrence(path_recurrence: str) -> bool:
    """
    Check if the recurrence annotation file exists and is not empty.

    :param path_recurrence: Path to the recurrence annotation file.
    :return: True if the file exists and is not empty, False otherwise.
    """
    try:
        with open(path_recurrence, 'r') as f:
            data = f.read().strip()
            return data == 'True'
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Recurrence annotation file not found at {path_recurrence}")


def df_equal(df1: pd.DataFrame, df2: pd.DataFrame, drop_last_n: int) -> bool:
    """
    Return True if two DataFrames have the same values, dtypes, indices, and columns.
    """

    if drop_last_n > 0:
        df2 = df2.iloc[:, :-drop_last_n]

    try:
        assert_frame_equal(df1, df2, check_dtype=True, check_like=True)
        return True
    except AssertionError:
        return False
