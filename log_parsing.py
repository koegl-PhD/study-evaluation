from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_log_to_df(file_path: str) -> pd.DataFrame:
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


def extract_pos_2d(df: pd.DataFrame) -> pd.DataFrame:
    """Extract x, y coordinates from detail field."""
    pos = df['detail'].str.extract(
        r'\(([-+]?\d+),\s*([-+]?\d+)\)').astype(float)
    return pos.rename(columns={0: 'x', 1: 'y'})


def extract_d(df: pd.DataFrame) -> pd.Series:
    """Extract d value from detail field."""
    return df['detail'].str.extract(r'd=([-+]?[0-9]*\.?[0-9]+)').astype(float)


def compute_slider_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Compute slider scroll stats."""
    df = df[df['action'] == 'Slider_Scroll']
    df = df[df['detail'].notnull()]
    df['position'] = df['detail'].str.extract(
        r'pos=([-+]?[0-9]*\.?[0-9]+)').astype(float)
    df['position_shifted'] = df.groupby(group_keys)['position'].shift(1)
    df['distance_delta'] = (df['position'] - df['position_shifted']).abs()
    return df.groupby(group_keys).agg(
        slider_total_distance=('distance_delta', 'sum'),
        slider_usage_count=('position', 'count')
    ).reset_index()


def compute_wheel_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Compute wheel scroll stats."""
    df = df[df['action'] == 'Wheel_Scroll']
    df = df[df['detail'].notnull()]
    df['delta'] = df['detail'].str.extract(r'd=([-+]?\d+)').astype(float)
    stats = df.groupby(group_keys)['delta'].value_counts().unstack(
        fill_value=0).reset_index()
    return stats.rename(columns={-1.0: 'wheel_scroll_d_-1_count', 1.0: 'wheel_scroll_d_1_count'})


def compute_positional_stats(df: pd.DataFrame, task: str, group_keys: list[str]) -> pd.DataFrame:
    """Compute distance and count for a positional task including Zoom."""
    sub = df[df['action'].str.startswith(task)].copy()
    pos = extract_pos_2d(sub)
    sub = sub.join(pos)
    sub[['x_prev', 'y_prev']] = sub.groupby(group_keys)[['x', 'y']].shift(1)
    sub['distance'] = np.sqrt(
        (sub['x'] - sub['x_prev'])**2 + (sub['y'] - sub['y_prev'])**2)
    return sub.groupby(group_keys).agg(
        **{f'{task.lower()}_total_distance': ('distance', 'sum'),
           f'{task.lower()}_usage_count': ('x', 'count')}
    ).reset_index()


def compute_zoom_d_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Compute zoom d-change and d-distance."""
    sub = df[df['action'].str.startswith('Zoom')].copy()
    sub['d_value'] = extract_d(sub)
    sub['d_prev'] = sub.groupby(group_keys)['d_value'].shift(1)
    sub['d_delta'] = sub['d_value'] - sub['d_prev']
    return sub.groupby(group_keys).agg(
        zoom_total_d_change=('d_value', 'sum'),
        zoom_total_d_distance=('d_delta', lambda x: x.abs().sum())
    ).reset_index()


def compute_double_click_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """
    Compute double click counts per predefined view.
    """

    # select only double-click events
    sub = df[df['action'] == 'Double click'].copy()

    # treat detail as view identifier
    sub['view'] = sub['detail']

    # count occurrences per group and view
    stats = sub.groupby(group_keys + ['view']).size().unstack(fill_value=0)

    # ensure zero columns for all expected views
    expected = ['Red1', 'Red2', 'Green1', 'Green2', 'Yellow1', 'Yellow2']
    for v in expected:
        if v not in stats.columns:
            stats[v] = 0

    # order columns and rename to include count suffix
    stats = stats[expected]
    stats.columns = [f'double_click_{v}_count' for v in expected]

    return stats.reset_index()


def compute_sync_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Compute count of user synchronised views."""
    sub = df[df['action'] == 'User synchronised views']
    return sub.groupby(group_keys).size().reset_index(name='synchronised_count')


def compute_arrow_key_stats(df: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    """Compute total arrow key presses per predefined view."""
    sub = df[df['action'] == 'Key_Arrow'].copy()
    sub['view'] = sub['detail'].str.split('~').str[0].str.strip()
    stats = sub.groupby(group_keys + ['view']).size().unstack(fill_value=0)
    expected = ['Red1', 'Red2', 'Green1', 'Green2', 'Yellow1', 'Yellow2']
    for v in expected:
        if v not in stats.columns:
            stats[v] = 0
    stats = stats[expected]
    stats.columns = [f'arrow_key_{v}_count' for v in expected]
    return stats.reset_index()


def aggregate_interaction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute combined scroll, mouse, arrow-key, and sync stats per task/view.
    """

    # define grouping keys and sort by timestamp
    group_keys = ['user_id', 'patient_id',
                  'transform_type', 'task_id', 'task_index']
    df_sorted = df.sort_values(by=group_keys + ['timestamp'])

    # compute slider and wheel stats
    slider_stats = compute_slider_stats(df_sorted, group_keys)
    wheel_stats = compute_wheel_stats(df_sorted, group_keys)

    # filter mouse events with valid details or coordinates
    base_mouse = df_sorted[df_sorted['event_type'] == 'U_MOUSE']
    base_mouse = base_mouse[
        base_mouse['detail'].notnull() |
        base_mouse['action'].str.contains(r'\(\d+,')
    ]

    # compute positional, zoom, double-click, arrow-key, and sync stats
    tasks = ['Pan', 'Zoom', 'Window_Level', 'Drag_Scroll']
    positional_stats = [compute_positional_stats(
        base_mouse, t, group_keys) for t in tasks]
    zoom_d_stats = compute_zoom_d_stats(base_mouse, group_keys)
    double_click_stats = compute_double_click_stats(base_mouse, group_keys)
    arrow_stats = compute_arrow_key_stats(df_sorted, group_keys)
    sync_stats = compute_sync_stats(df_sorted, group_keys)

    # start merge with unique base keys
    base_keys = df[group_keys].drop_duplicates()
    out = base_keys.merge(slider_stats, on=group_keys, how='left') \
                   .merge(wheel_stats, on=group_keys, how='left')
    # merge all positional stats
    for stat in positional_stats:
        out = out.merge(stat, on=group_keys, how='left')
    # merge remaining stats
    out = out.merge(zoom_d_stats, on=group_keys, how='left') \
             .merge(double_click_stats, on=group_keys, how='left') \
             .merge(arrow_stats, on=group_keys, how='left') \
             .merge(sync_stats, on=group_keys, how='left')

    # fill missing with zeros and exclude training data
    out.fillna(0, inplace=True)
    out = out[~out['patient_id'].str.contains('training')]

    # verify row count
    if len(out) != 240:
        raise ValueError(
            f"Expected 240 rows, got {len(out)}. Check input data.")

    # reset index and compute combined wheel scroll distance
    out.reset_index(drop=True, inplace=True)
    out['wheel_scroll_distance_c'] = out['wheel_scroll_d_-1_count'] + \
        out['wheel_scroll_d_1_count']

    # drop intermediate columns and rename distances
    out.drop(columns=[
        'wheel_scroll_d_-1_count', 'wheel_scroll_d_1_count',
        'zoom_total_d_change', 'zoom_total_d_distance'
    ], inplace=True)
    out.rename(columns={
        'slider_total_distance': 'slider_total_distance_mm',
        'pan_total_distance': 'pan_total_distance_px',
        'zoom_total_distance': 'zoom_total_distance_px',
        'window_level_total_distance': 'window_level_total_distance_px',
        'drag_scroll_total_distance': 'drag_scroll_total_distance_px'
    }, inplace=True)

    # reorder columns: base, wheel dist, double-clicks, sync, then arrow-keys
    base_cols = [c for c in out.columns
                 if not c.startswith(('double_click_', 'arrow_key_', 'synchronised_count'))
                 and c != 'wheel_scroll_distance_c']
    expected = ['Red1', 'Red2', 'Green1', 'Green2', 'Yellow1', 'Yellow2']
    dbl_cols = [f'double_click_{v}_count' for v in expected]
    arrow_cols = [c for c in out.columns if c.startswith('arrow_key_')]
    out = out[base_cols + ['wheel_scroll_distance_c'] +
              dbl_cols + ['synchronised_count'] + arrow_cols]

    return out


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


def extract_window_adjust_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and pair window level adjustment start and end events."""
    start_df = df[df['action'] == 'Start Window_Level'][
        ['user_id', 'patient_id', 'transform_type',
            'task_id', 'task_index', 'timestamp']
    ].copy()
    end_df = df[df['action'] == 'End Window_Level'][
        ['user_id', 'patient_id', 'transform_type',
            'task_id', 'task_index', 'timestamp']
    ].copy()
    start_df = start_df.rename(columns={'timestamp': 'adjust_start'})
    end_df = end_df.rename(columns={'timestamp': 'adjust_end'})
    return pd.merge_asof(
        start_df.sort_values('adjust_start'),
        end_df.sort_values('adjust_end'),
        by=['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'],
        left_on='adjust_start',
        right_on='adjust_end',
        direction='forward'
    )


def sum_window_adjust_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Sum window level adjustment durations per task."""
    df['adjust_duration'] = (
        df['adjust_end'] - df['adjust_start']).dt.total_seconds()
    totals = df.groupby(
        ['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index']
    )['adjust_duration'].sum().reset_index()
    return totals.rename(columns={'adjust_duration': 'total_adjust_seconds'})


def compute_task_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute net task durations excluding pauses and window adjustments."""
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], format="%Y-%m-%d %H:%M:%S,%f", errors='coerce'
    )

    # start/end extraction
    duration_df = extract_task_start_end_times(df)

    # pause durations
    pause_pairs = extract_pause_resume_pairs(df)
    pause_sums = sum_pause_durations(pause_pairs)
    duration_df = pd.merge(
        duration_df, pause_sums,
        on=['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'],
        how='left'
    ).fillna({'total_pause_seconds': 0})

    # window adjustment durations
    adjust_pairs = extract_window_adjust_pairs(df)
    adjust_sums = sum_window_adjust_durations(adjust_pairs)
    duration_df = pd.merge(
        duration_df, adjust_sums,
        on=['user_id', 'patient_id', 'transform_type', 'task_id', 'task_index'],
        how='left'
    ).fillna({'total_adjust_seconds': 0})

    # compute net duration
    duration_df['duration_seconds'] = (
        duration_df['end_time'] - duration_df['start_time']
    ).dt.total_seconds() \
        - duration_df['total_pause_seconds'] \
        - duration_df['total_adjust_seconds']

    # tidy and filter
    duration_df = duration_df[~duration_df['patient_id'].str.contains(
        'training')]
    return duration_df.drop(
        columns=['start_time', 'end_time',
                 'total_pause_seconds', 'total_adjust_seconds']
    )
