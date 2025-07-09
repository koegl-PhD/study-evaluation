import glob
import json

from typing import Any, Dict, List, LiteralString

import numpy as np
import pandas as pd


def get_rt_bifurcation_paths(path_rt: str, patient_id: str) -> List[str]:

    path_patient = f"{path_rt}/{patient_id}/*/"

    files = glob.glob(path_patient + "*a_*.mrk.json")

    if len(files) != 4:
        raise FileNotFoundError(
            f"Expected 4 bifurcation files for patient {patient_id}, found {len(files)}")

    files.sort()

    return files


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


def insert_bifurcations(df_duration: pd.DataFrame, path_gt: str, path_rt: str) -> pd.DataFrame:
    # has to be the df_duration

    columns_duration: list[LiteralString] = "user_id patient_id task_id task_index transform_type duration_seconds".split()

    if df_duration.columns.tolist() != columns_duration:
        raise ValueError(
            f"df_duration columns {df_duration.columns.tolist()} do not match expected {columns_duration}")

    points_rt = get_rt_bifurcation_locations(path_rt, r"0f9SYhwcPFc")
    points_gt = get_gt_bifurcation_locations(path_gt, r"0f9SYhwcPFc")

    x = 0
