import glob
import json

from typing import Any, Dict, List, LiteralString

import numpy as np
import pandas as pd

import utils


def get_rt_bifurcation_paths(path_rt: str, patient_id: str) -> List[str]:

    path_patient = f"{path_rt}/{patient_id}/*/"

    files = glob.glob(path_patient + "*a_*.mrk.json")

    if len(files) != 4:
        raise FileNotFoundError(
            f"Expected 4 bifurcation files for patient {patient_id}, found {len(files)}")

    files.sort()

    return files


def get_rt_lymphnode_path(path_rt: str, patient_id: str) -> str:
    path_patient = f"{path_rt}/{patient_id}/*/"

    files = glob.glob(path_patient + "*lymph_node_*.mrk.json")

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected 1 lymph node file for patient {patient_id}, found {len(files)}")

    return files[0]


def get_gt_bifurcation_path(path_gt: str, patient_id: str) -> str:

    categories = glob.glob(path_gt + "/*")
    for category in categories:
        patients = glob.glob(category + "/*")

        for patient in patients:
            if patient_id in patient:
                path = patient + "/preprocessed"

                folders_studies = glob.glob(path + "/*")
                folders_studies.sort()
                study_b = folders_studies[1]

                path_points = glob.glob(
                    study_b + "/annotations/*points*.mrk.json")
                if len(path_points) != 1:
                    raise FileNotFoundError(
                        f"Expected 1 bifurcation points file for patient {patient_id}, found {len(path_points)}")
                return path_points[0]

    raise FileNotFoundError(
        f"Patient {patient_id} not found in {path_gt}")


def get_gt_lymphnode_path(path_gt: str, patient_id: str) -> str:
    categories = glob.glob(path_gt + "/*")
    for category in categories:
        patients = glob.glob(category + "/*")

        for patient in patients:
            if patient_id in patient:
                path = patient + "/preprocessed"

                folders_studies = glob.glob(path + "/*")
                folders_studies.sort()
                study_b = folders_studies[1]

                path_lymphnodes = glob.glob(
                    study_b + "/annotations/roi_lymphnode*.mrk.json")
                if len(path_lymphnodes) != 1:
                    raise FileNotFoundError(
                        f"Expected 1 lymph node file for patient {patient_id}, found {len(path_points)}")
                return path_lymphnodes[0]

    raise FileNotFoundError(
        f"Patient {patient_id} not found in {path_gt}")


def get_rt_bifurcation_locations(path_rt: str, patient_id: str) -> Dict[str, np.ndarray[Any, Any]]:
    """
    Get bifurcation locations from the RT files.
    :param path_rt: Path to the RT files.
    :param patient_id: Patient ID to filter the files.
    :return: List of bifurcation locations as numpy arrays.
    """
    paths = get_rt_bifurcation_paths(path_rt, patient_id)

    bifurcations = {}
    for path in paths:
        bifurcation_name: str = '_'.join(path.split('/')[-1].split('_')[:3])

        with open(path, 'r') as f:
            data = f.read()
            data_json = json.loads(data)

            point = data_json['markups'][0]['controlPoints'][0]['position']

            bifurcations[bifurcation_name] = np.array(point)

    return bifurcations


def get_rt_lymphnode_location(path_rt: str, patient_id: str) -> np.ndarray[Any, Any]:
    """
    Get lymph node location from the RT files.
    :param path_rt: Path to the RT files.
    :param patient_id: Patient ID to filter the files.
    :return: Lymph node location as a numpy array.
    """
    path = get_rt_lymphnode_path(path_rt, patient_id)

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        position = data_json['markups'][0]['controlPoints'][0]['position']

    return np.array(position)


def get_gt_bifurcation_locations(path_gt: str, patient_id: str) -> Dict[str, np.ndarray[Any, Any]]:
    """
    Get bifurcation locations from the GT files.
    :param path_gt: Path to the GT files.
    :param patient_id: Patient ID to filter the files.
    :return: List of bifurcation locations as numpy arrays.
    """
    path = get_gt_bifurcation_path(path_gt, patient_id)

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        points = data_json['markups'][0]['controlPoints']

        bifurcations = {}
        for point in points:
            bifurcation_name = '_'.join(point['label'].split('_')[1:4])
            position = point['position']
            bifurcations[bifurcation_name] = np.array(position)

    return bifurcations


def get_gt_lymphnode(path_gt: str, patient_id: str) -> Dict[str, np.ndarray[Any, Any]]:
    """
    Get lymph node location from the GT files.
    :param path_gt: Path to the GT files.
    :param patient_id: Patient ID to filter the files.
    :return: Lymph node location as a numpy array.
    """
    path = get_gt_lymphnode_path(path_gt, patient_id)

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        center = data_json['markups'][0]['center']
        size = data_json['markups'][0]['size']

    return {"center": np.array(center), "size": np.array(size)}


def insert_bifurcations(
        df_duration: pd.DataFrame,
        path_gt: str,
        path_rt: str,
        tolerance: float) -> pd.DataFrame:
    # has to be the df_duration

    df_duration['result_abs'] = None
    df_duration['result_rel'] = None

    columns_duration: list[LiteralString] = "user_id patient_id task_id task_index transform_type duration_seconds result_abs result_rel".split()

    if df_duration.columns.tolist() != columns_duration:
        raise ValueError(
            f"df_duration columns {df_duration.columns.tolist()} do not match expected {columns_duration}")

    for index, row in df_duration.iterrows():

        t = str(row['task_id'])

        if row['task_id'] in ['a_vertebralis_r', 'a_vertebralis_l', 'a_carotisexterna_r', 'a_carotisexterna_l']:
            points_rt = get_rt_bifurcation_locations(
                path_rt, row['patient_id'])
            points_gt = get_gt_bifurcation_locations(
                path_gt, row['patient_id'])

            norm = np.linalg.norm(points_gt[t] - points_rt[t])
            df_duration.at[index, 'result_rel'] = norm
            df_duration.at[index, 'result_abs'] = norm < tolerance

    return df_duration


def insert_lymphnodes(
        df_duration: pd.DataFrame,
        path_gt: str,
        path_rt: str) -> pd.DataFrame:
    """
    Insert lymph node locations into the DataFrame.
    :param df_duration: DataFrame with task durations.
    :param path_gt: Path to the ground truth files.
    :param path_rt: Path to the RT files.
    :return: DataFrame with lymph node locations.
    """
    df_duration['lymph_node_abs'] = None
    df_duration['lymph_node_rel'] = None

    for index, row in df_duration.iterrows():

        task = str(row['task_id'])

        if task == "lymph_node":
            lymphnode_center_rt = get_rt_lymphnode_location(
                path_rt, row['patient_id'])
            lymph_node_gt = get_gt_lymphnode(
                path_gt, row['patient_id'])

            # check if lymphnode_center_rt is inside the GT lymph node
            center = lymph_node_gt['center']
            size = lymph_node_gt['size']
            df_duration.at[index, 'lymph_node_abs'] = utils.is_point_in_ROI(
                lymphnode_center_rt, center, size)
            df_duration.at[index, 'lymph_node_rel'] = np.linalg.norm(
                lymphnode_center_rt - center)

    return df_duration


def insert_study_results(
    df: pd.DataFrame,
    path_gt: str,
    path_rt: str,
    tolerance_bifurcations: float
) -> pd.DataFrame:

    df = insert_bifurcations(df, path_gt, path_rt, tolerance_bifurcations)
    df = insert_lymphnodes(df, path_gt, path_rt)

    return df
