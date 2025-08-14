import glob
import json

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

import utils

####################################################################################
####################  PATHS  #######################################################
####################################################################################
#### RT PATHS ####


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


def get_rt_recurrence_path(path_rt: str, patient_id: str) -> Optional[str]:
    path_patient = f"{path_rt}/{patient_id}/*/"

    file_recurrence_txt = glob.glob(path_patient + "recurrence_present.txt")[0]

    if not utils.did_rad_check_recurrence(file_recurrence_txt):
        return None

    files = glob.glob(path_patient + "*recurrence_*.mrk.json")

    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected 1 recurrence file for patient {patient_id}, found {len(files)}")

    return files[0]

#### GT PATHS ####


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


def get_gt_lymphnode_path(path_gt: str, patient_id: str) -> Dict[str, str]:
    categories = glob.glob(path_gt + "/*")
    for category in categories:
        patients = glob.glob(category + "/*")

        for patient in patients:
            if patient_id in patient:
                path = patient + "/preprocessed"

                folders_studies = glob.glob(path + "/*")
                folders_studies.sort()

                study_a = folders_studies[0]
                study_b = folders_studies[1]

                path_lymphnodes_a = glob.glob(
                    study_a + "/annotations/roi_lymphnode*.mrk.json")
                path_lymphnodes_b = glob.glob(
                    study_b + "/annotations/roi_lymphnode*.mrk.json")
                if len(path_lymphnodes_b) != 1 or len(path_lymphnodes_a) != 1:
                    raise FileNotFoundError(
                        f"Expected 1 lymph node file for patient {patient_id}, found {len(path_lymphnodes_a)} in study A and {len(path_lymphnodes_b)} in study B")

                return {
                    "a": path_lymphnodes_a[0],
                    "b": path_lymphnodes_b[0]
                }

    raise FileNotFoundError(
        f"Patient {patient_id} not found in {path_gt}")


def get_gt_recurrence_path(path_gt: str, patient_id: str) -> Optional[str]:
    categories = glob.glob(path_gt + "/*")
    for category in categories:
        patients = glob.glob(category + "/*")

        for patient in patients:
            if patient_id in patient:
                path = patient + "/preprocessed"

                folders_studies = glob.glob(path + "/*")
                folders_studies.sort()
                study_b = folders_studies[1]

                path_recurrence = glob.glob(
                    study_b + "/annotations/roi_recurrence*.mrk.json")
                if len(path_recurrence) == 0:
                    return None
                elif len(path_recurrence) > 1:
                    raise FileExistsError(
                        f"Multiple recurrence files found for patient {patient_id}")
                return path_recurrence[0]

    return None


####################################################################################
####################  LOCATIONS  ###################################################
####################################################################################
#### RT LOCATIONS ####
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


def get_rt_recurrence_location(path_rt: str, patient_id: str) -> Optional[np.ndarray[Any, Any]]:
    """
    Get recurrence location from the RT files.
    :param path_rt: Path to the RT files.
    :param patient_id: Patient ID to filter the files.
    :return: Recurrence location as a numpy array, or None if no recurrence is present.
    """
    path = get_rt_recurrence_path(path_rt, patient_id)

    if path is None:
        return None

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        position = data_json['markups'][0]['controlPoints'][0]['position']

    return np.array(position)

#### GT LOCATIONS ####


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
    path = get_gt_lymphnode_path(path_gt, patient_id)['b']

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        center = data_json['markups'][0]['center']
        size = data_json['markups'][0]['size']

    return {"center": np.array(center), "size": np.array(size)}


def get_gt_recurrence(path_gt: str, patient_id: str) -> Optional[Dict[str, np.ndarray[Any, Any]]]:
    """
    Get recurrence location from the GT files.
    :param path_gt: Path to the GT files.
    :param patient_id: Patient ID to filter the files.
    :return: Recurrence location as a numpy array, or None if no recurrence is present.
    """
    path = get_gt_recurrence_path(path_gt, patient_id)

    if path is None:
        return None

    with open(path, 'r') as f:
        data = f.read()
        data_json = json.loads(data)

        roi = data_json['markups'][0]
        center = np.array(roi['center'])
        size = np.array(roi['size'])

        return {'center': center, 'size': size}


def insert_bifurcations(
        df: pd.DataFrame,
        path_gt: str,
        path_rt: str,
        tolerance: float
) -> pd.DataFrame:

    task_ids = [
        "a_vertebralis_r",
        "a_vertebralis_l",
        "a_carotisexterna_r",
        "a_carotisexterna_l"
    ]

    for task_id in task_ids:
        name_rel = f"{task_id}_rel"
        name_abs = f"{task_id}_abs_{tolerance}"

        df[name_rel] = None
        df[name_abs] = None

        for index, row in df.iterrows():

            if row['task_id'] == task_id:

                points_rt = get_rt_bifurcation_locations(
                    path_rt, row['patient_id'])
                points_gt = get_gt_bifurcation_locations(
                    path_gt, row['patient_id'])

                norm = np.linalg.norm(points_gt[task_id] - points_rt[task_id])
                df.at[index, name_rel] = norm
                df.at[index, name_abs] = norm < tolerance

    return df


def insert_lymphnodes(
        df: pd.DataFrame,
        path_gt: str,
        path_rt: str
) -> pd.DataFrame:
    """
    Insert lymph node locations into the DataFrame. If rt point is inside the GT lymph node, set lymph_node_abs to True, otherwise False.
    :param df: DataFrame with task durations.
    :param path_gt: Path to the ground truth files.
    :param path_rt: Path to the RT files.
    :return: DataFrame with lymph node locations.
    """

    df['lymph_node_abs'] = None

    for index, row in df.iterrows():

        task = str(row['task_id'])

        if task == "lymph_node":
            lymphnode_center_rt = get_rt_lymphnode_location(
                path_rt, row['patient_id'])
            lymph_node_gt = get_gt_lymphnode(
                path_gt, row['patient_id'])

            # check if lymphnode_center_rt is inside the GT lymph node
            center = lymph_node_gt['center']
            size = lymph_node_gt['size']
            df.at[index, 'lymph_node_abs'] = utils.is_point_in_ROI(
                lymphnode_center_rt, center, size)

    insert_lymphnodes_tre(df)

    return df


def insert_lymphnodes_tre(
        df: pd.DataFrame
) -> pd.DataFrame:

    path_lymph_tres = "tres_lymph_node.csv"
    df_tres = pd.read_csv(path_lymph_tres)

    df['lymph_node_tre_linear'] = None
    df['lymph_node_tre_nonlinear'] = None

    for index, row in df.iterrows():
        task = str(row['task_id'])

        if task == "lymph_node":

            patient_id = row['patient_id']
            lymph_node_tre = df_tres[df_tres['patient_id'] == patient_id]

            if lymph_node_tre.empty:
                raise ValueError(
                    f"No TRE values found for patient {patient_id} in {path_lymph_tres}")

            df.at[index, 'lymph_node_tre_linear'] = lymph_node_tre['tre_lin'].values[0]
            df.at[index, 'lymph_node_tre_nonlinear'] = lymph_node_tre['tre_def'].values[0]


def insert_recurrence(
        df: pd.DataFrame,
        path_gt: str,
        path_rt: str
) -> pd.DataFrame:

    df['recurrence'] = None
    df['recurrence'] = pd.Categorical(
        df['recurrence'], categories=['tp', 'fp', 'tn', 'fn'])

    for index, row in df.iterrows():
        if row['task_id'] == 'recurrence':
            recurrence_rt = get_rt_recurrence_location(
                path_rt, row['patient_id'])
            recurrence_gt = get_gt_recurrence(
                path_gt, row['patient_id'])

            value = None

            if recurrence_gt is None and recurrence_rt is None:
                value = 'tn'
            elif recurrence_gt is None and recurrence_rt is not None:
                value = 'fp'
            elif recurrence_gt is not None and recurrence_rt is None:
                value = 'fn'
            elif recurrence_gt is not None and recurrence_rt is not None:

                # check if recurrence_rt is inside the GT recurrence ROI
                center = recurrence_gt['center']
                size = recurrence_gt['size']

                correct = utils.is_point_in_ROI(
                    recurrence_rt, center, size)

                value = 'tp' if correct else 'fp'

            df.at[index, 'recurrence'] = value

    return df


def insert_study_results(
    df: pd.DataFrame,
    path_gt: str,
    path_rt: str,
    tolerance_bifurcations: float,
    rad_contents: Dict[str, int | bool | str]
) -> pd.DataFrame:

    df = insert_bifurcations(df, path_gt, path_rt, tolerance_bifurcations)

    df = insert_lymphnodes(df, path_gt, path_rt)

    df = insert_recurrence(df, path_gt, path_rt)

    df.insert(loc=1, column="group", value=rad_contents['group'])
    df.insert(loc=1, column="experienced",
              value=rad_contents['experienced'])

    return df
