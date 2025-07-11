from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_log_v2_to_df(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        rows: list[Dict[str, Any] | None] = []
        for line in file:
            parsed = parse_log_line_v2(line)
            if parsed:
                rows.append(parsed)

    rows = clear_user_started_study(rows)

    rows = clear_specific(rows, 'action', 'Organiser started study')
    rows = clear_specific(rows, 'action', 'UI set to simple')
    rows = clear_specific(rows, 'action', 'All chunks')
    rows = clear_specific(rows, 'action', 'Chunk ')
    rows = clear_specific(rows, 'action', '\t')
    rows = clear_specific(rows, 'action', 'All tasks to be done')
    rows = clear_specific(rows, 'action', 'Combination')
    rows = clear_specific(rows, 'action', 'Start loading study data chunk')
    rows = clear_specific(rows, 'action', 'Finished loading study data chunk')
    rows = clear_specific(rows, 'action', 'User closed study description')
    rows = clear_specific(
        rows, 'action', 'User closed training study description')
    rows = clear_specific(rows, 'action', 'Start task')
    rows = clear_specific(rows, 'action', 'User started next patient')
    rows = clear_specific(rows, 'action', 'Point saved')
    rows = clear_specific(rows, 'action', 'No recurrence to save')
    rows = clear_specific(
        rows, 'action', 'Start clearing current study data chunk')
    rows = clear_specific(
        rows, 'action', 'Done clearing current study data chunk')
    rows = clear_specific(rows, 'action', 'User closed study begins')

    return pd.DataFrame(rows)


def clear_user_started_study(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:

    number_of_occurrences = sum(
        1 for row in rows if row['action'].startswith('User started study'))

    for _ in range(number_of_occurrences):
        index_started_study = -1
        for i, row in enumerate(rows):
            if row['action'].startswith('User started study'):
                index_started_study = i
                break

        index_closed_info_popup = -1
        for i, row in enumerate(rows):
            if row['action'].startswith('User closed info popup'):
                index_closed_info_popup = i
                break
        if index_started_study == -1 or index_closed_info_popup == -1:
            raise ValueError(
                "Could not find 'User started study' or 'User closed info popup' in rows"
            )

        if index_started_study > index_closed_info_popup:
            raise ValueError(
                f"Index of 'User started study' ({index_started_study}) must be less than index of 'User closed info popup' ({index_closed_info_popup})"
            )

        rows = rows[:index_started_study] + rows[index_closed_info_popup + 1:]

    return rows


def clear_specific(rows: List[Dict[str, str]], content_type: str, content: str) -> List[Dict[str, str]]:
    """
    Clear all rows that contain a specific content in a specific column.
    """
    return [row for row in rows if not row[content_type].startswith(content)]


def parse_log_line_v2(line: str) -> Optional[Dict[str, Any]]:
    parts = line.strip().split(" ~ ")
    if len(parts) < 10:
        return None

    timestamp = parts[0]
    module = parts[1]
    log_level = parts[2]
    event_type = parts[3]
    task_index = parts[4]
    user_id = parts[5]
    patient_id = parts[6]
    transform_type = parts[7]
    task_id = parts[8]
    action = parts[9]
    detail = " ~ ".join(parts[10:]) if len(parts) > 10 else None

    return {
        "timestamp": timestamp,
        "module": module,
        "log_level": log_level,
        "event_type": event_type,
        "task_index": task_index,
        "user_id": user_id,
        "patient_id": patient_id,
        "transform_type": transform_type,
        "task_id": task_id,
        "action": action,
        "detail": detail
    }


def compute_scroll_stats_grouped_v2(df: pd.DataFrame) -> pd.DataFrame:
    # --- Slider Scroll ---
    slider_df = df[df['action'] == 'Slider_Scroll'].copy()
    slider_df = slider_df[slider_df['detail'].notnull()]
    slider_df['position'] = slider_df['detail'].str.extract(
        r'pos=([-+]?[0-9]*\.?[0-9]+)').astype(float)

    # Sort and compute pairwise distance per group
    slider_df.sort_values(by=['user_id', 'patient_id', 'transform_type',
                          'task_id', 'task_index', 'timestamp'], inplace=True)
    slider_df['position_shifted'] = slider_df.groupby(
        ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'])['position'].shift(1)
    slider_df['distance_delta'] = (
        slider_df['position'] - slider_df['position_shifted']).abs()

    slider_stats = slider_df.groupby(['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index']).agg(
        slider_total_distance=('distance_delta', 'sum'),
        slider_usage_count=('position', 'count')
    ).reset_index()

    # --- Wheel Scroll ---
    wheel_df = df[df['action'] == 'Wheel_Scroll'].copy()
    wheel_df = wheel_df[wheel_df['detail'].notnull()]
    wheel_df['delta'] = wheel_df['detail'].str.extract(
        r'd=([-+]?[0-9]+)').astype(float)

    wheel_stats = wheel_df.groupby(['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'])[
        'delta'].value_counts().unstack(fill_value=0).reset_index()
    wheel_stats.rename(columns={-1.0: 'wheel_scroll_d_-1_count',
                       1.0: 'wheel_scroll_d_1_count'}, inplace=True)

    # Merge both stats
    keys = ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index']
    base_df = df[keys].drop_duplicates()

    combined_stats = (
        base_df
        .merge(slider_stats, how='left', on=keys)
        .merge(wheel_stats,  how='left', on=keys)
        .fillna(0)
    )

    combined_stats = combined_stats[~combined_stats['patient_id'].str.contains(
        'training')]

    if len(combined_stats) != 240:
        raise ValueError(
            f"Expected 240 rows, got {len(combined_stats)}. Check the input data.")

    combined_stats = combined_stats.reset_index(drop=True)

    return combined_stats


def compute_task_scroll_stats_v2(df: pd.DataFrame) -> pd.DataFrame:
    # --- Helper to extract 2D coordinates from either format ---
    def extract_pos_2d(df):
        pos_xy = df['detail'].str.extract(r'\(([-+]?\d+),\s*([-+]?\d+)\)')
        return pos_xy.astype(float).rename(columns={0: 'x', 1: 'y'})

    # Extract d from detail field
    def extract_d(df):
        return df['detail'].str.extract(r'd=([-+]?[0-9]*\.?[0-9]+)').astype(float)

    # Filter for mouse events with usable detail or coordinate string
    base_df = df[df['event_type'] == 'U_MOUSE'].copy()
    base_df = base_df[base_df['detail'].notnull(
    ) | base_df['action'].str.contains(r'\(\d+,')]

    result_frames = []
    task_types = ['Pan', 'Zoom', 'Window_Level', 'Drag_Scroll']

    for task in task_types:
        task_df = base_df[base_df['action'].str.startswith(task)].copy()
        pos_df = extract_pos_2d(task_df)
        task_df = task_df.join(pos_df)

        task_df.sort_values(by=['user_id', 'patient_id', 'transform_type',
                            'task_id', 'task_index', 'timestamp'], inplace=True)
        task_df[['x_prev', 'y_prev']] = task_df.groupby(
            ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'])[['x', 'y']].shift(1)
        task_df['distance'] = np.sqrt(
            (task_df['x'] - task_df['x_prev'])**2 + (task_df['y'] - task_df['y_prev'])**2)

        agg = task_df.groupby(['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index']).agg(
            **{f'{task.lower()}_total_distance': ('distance', 'sum'),
               f'{task.lower()}_usage_count': ('x', 'count')}
        ).reset_index()

        result_frames.append(agg)

    # --- Zoom-specific: sum of d and d-distance ---
    zoom_df = base_df[base_df['action'].str.startswith('Zoom')].copy()
    zoom_df['d_value'] = extract_d(zoom_df)
    zoom_df.sort_values(by=['user_id', 'patient_id', 'transform_type',
                        'task_id', 'task_index', 'timestamp'], inplace=True)
    zoom_df['d_prev'] = zoom_df.groupby(
        ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'])['d_value'].shift(1)
    zoom_df['d_delta'] = zoom_df['d_value'] - zoom_df['d_prev']

    zoom_d_sum = zoom_df.groupby(['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'])[
        'd_value'].sum().reset_index()
    zoom_d_sum.rename(columns={'d_value': 'zoom_total_d_change'}, inplace=True)

    zoom_d_dist = zoom_df['d_delta'].abs().groupby([
        zoom_df['user_id'],
        zoom_df['patient_id'],
        zoom_df['transform_type'],
        zoom_df['task_id'],
        zoom_df['task_index']
    ]).sum().reset_index()
    zoom_d_dist.rename(
        columns={'d_delta': 'zoom_total_d_distance'}, inplace=True)

    # Merge all statistics
    keys = ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index']
    base_df = df[keys].drop_duplicates()

    merged = base_df
    for stats in result_frames:
        merged = merged.merge(stats, on=keys, how='left')

    merged = merged.merge(zoom_d_sum,  on=keys, how='left')
    merged = merged.merge(zoom_d_dist, on=keys, how='left')

    merged.fillna(0, inplace=True)

    merged = merged[~merged['patient_id'].str.contains(
        'training')]

    if len(merged) != 240:
        raise ValueError(
            f"Expected 240 rows, got {len(merged)}. Check the input data.")

    merged = merged.reset_index(drop=True)

    return merged


def extract_task_start_end_times(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts start and end times for each task."""
    task_control_df = df[(df['event_type'] == 'U_BUTTON') & (
        df['action'].isin(['User started task', 'Next task']))].copy()
    start_df = task_control_df[task_control_df['action']
                               == 'User started task'].copy()
    end_df = task_control_df[task_control_df['action'] == 'Next task'].copy()
    start_df = start_df[['user_id', 'patient_id',
                         'transform_type', 'task_id', 'task_index', 'timestamp']]
    end_df = end_df[['user_id', 'patient_id', 'transform_type',
                     'task_id', 'task_index', 'timestamp']]
    start_df.rename(columns={'timestamp': 'start_time'}, inplace=True)
    end_df.rename(columns={'timestamp': 'end_time'}, inplace=True)
    duration_df = pd.merge(start_df, end_df, on=[
        'user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'], how='inner')
    return duration_df


def extract_pause_resume_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and pairs pause and resume events for each task, including both 'User paused study' and 'User clicked on info button' as pause events."""
    # Both types of pause events
    pause_df = df[(df['event_type'] == 'U_BUTTON') & (
        df['action'].isin(['User paused study', 'User clicked on info button'])
    )].copy()
    resume_df = df[(df['event_type'] == 'U_BUTTON') & (
        df['action'] == 'User resumed study')].copy()
    pause_df = pause_df[['user_id', 'patient_id',
                         'transform_type', 'task_id', 'task_index', 'timestamp']]
    resume_df = resume_df[['user_id', 'patient_id',
                           'transform_type', 'task_id', 'task_index', 'timestamp']]
    pause_df = pause_df.rename(columns={'timestamp': 'pause_time'})
    resume_df = resume_df.rename(columns={'timestamp': 'resume_time'})
    # Pair each pause with the next resume for the same task
    pause_resume_df = pd.merge_asof(
        pause_df.sort_values('pause_time'),
        resume_df.sort_values('resume_time'),
        by=['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'],
        left_on='pause_time',
        right_on='resume_time',
        direction='forward'
    )
    return pause_resume_df


def sum_pause_durations(pause_resume_df: pd.DataFrame) -> pd.DataFrame:
    """Sums pause durations for each task."""
    pause_resume_df['pause_duration'] = (
        pause_resume_df['resume_time'] - pause_resume_df['pause_time']).dt.total_seconds()
    pause_sums = pause_resume_df.groupby([
        'user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'
    ])['pause_duration'].sum().reset_index()
    pause_sums.rename(
        columns={'pause_duration': 'total_pause_seconds'}, inplace=True)
    return pause_sums


def compute_task_duration_by_index_v2(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format="%Y-%m-%d %H:%M:%S,%f", errors='coerce')

    # Extract start and end times
    duration_df = extract_task_start_end_times(df)

    # Extract pause/resume pairs and sum pause durations
    pause_resume_df = extract_pause_resume_pairs(df)
    pause_sums = sum_pause_durations(pause_resume_df)

    # Merge pause sums into duration_df
    duration_df = pd.merge(duration_df, pause_sums, on=[
                           'user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'], how='left')
    duration_df['total_pause_seconds'] = duration_df['total_pause_seconds'].fillna(
        0)

    # Compute net duration
    duration_df['duration_seconds'] = (
        duration_df['end_time'] - duration_df['start_time']).dt.total_seconds() - duration_df['total_pause_seconds']

    duration_df = duration_df.reindex(columns=[
        'user_id', 'patient_id', 'task_id', 'task_index',
        'transform_type', 'start_time', 'end_time',
        'total_pause_seconds', 'duration_seconds'])

    # remove columns start_time, end_time, total_pause_seconds
    duration_df = duration_df.drop(
        columns=['start_time', 'end_time', 'total_pause_seconds'])

    duration_df = duration_df[~duration_df['patient_id'].str.contains(
        'training')]

    return duration_df
