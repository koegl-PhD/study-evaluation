from datetime import datetime


import numpy as np


def elapsed_seconds(start: str, end: str) -> float:
    """Return elapsed seconds (float) between two datetime strings."""
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    t1 = datetime.strptime(start, fmt)
    t2 = datetime.strptime(end, fmt)
    return (t2 - t1).total_seconds()


path_log_niftyreg = r"/home/fryderyk/Downloads/log_niftyreg.log"
path_log_demons = r"/home/fryderyk/Downloads/log_demons.log"
path_log_syn = r"/home/fryderyk/Downloads/log_syn.log"

# niftyreg
with open(path_log_niftyreg, 'r') as file:

    lines = file.readlines()
    lines.sort()

    durations_set = {}

    for line in lines:
        if 'Running parameter set' in line:
            # extract the parameter set number
            param_set = line.split(' ')[-1].strip().split('/')[0]

            durations_set[param_set] = []

        if "Running NiftyReg in the terminal" in line:
            start = ' '.join(line.split(' ')[0:2])

        if "Transforming output to a displacement field" in line:
            end = ' '.join(line.split(' ')[0:2])
            duration = elapsed_seconds(start, end)

            durations_set.get(param_set, []).append(duration)

    x = 0

# demosn and syn
"""
with open(path_log_demons, 'r') as file:

    lines = file.readlines()
    lines.sort()

    durations_set = {}

    for line in lines:
        if 'Running parameter set' in line:
            # extract the parameter set number
            param_set = line.split(' ')[-1].strip().split('/')[0]

            durations_set[param_set] = []

        if "RegBaselines - 	Registering" in line:
            start = ' '.join(line.split(' ')[0:2])

        if "RegBaselines - 	Saving results" in line:
            end = ' '.join(line.split(' ')[0:2])
            duration = elapsed_seconds(start, end)

            durations_set.get(param_set, []).append(duration)

    x = 0
"""

# sum durations for each parameter set
durations_sum = {k: sum(v) for k, v in durations_set.items()}

durations = np.array(list(durations_sum.values()))

min_duration = durations.min()
max_duration = durations.max()
mean_duration = durations.mean()
std_duration = durations.std()

for k, v in durations_set.items():
    print(f"set: {k}, min: {min(v)}")

x = 0
