import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Data
data_all = np.array([
    [1.0, 0.000871, 7.193404e-11],
    [8.714404e-04, 1.0, 1.062370e-01],
    [7.193404e-11, 0.106237, 1.0]
])
data_inexperienced = np.array([
    [1.0, 0.001240, 1.831561e-05],
    [0.001240, 1.0, 0.902784],
    [1.831561e-05, 0.902784, 1.0]
])
data_experienced = np.array([
    [1.0, 0.222688, 0.000004],
    [0.222688, 1.0, 0.032508],
    [0.000004, 0.032508, 1.0]
])


labels = ["Good", "Moderate", "Poor"]

cmap = LinearSegmentedColormap.from_list("gray_red", ["#901818", "#dddddd"])
font_size = 13

mask = np.triu(np.ones((3, 3), dtype=bool), k=1)


def sci_fmt(x: float) -> str:
    """Format small numbers in exponential notation."""
    return f"{x:.1e}" if (x < 0.01 and x != 0) else f"{x:.3f}"


# Plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for data, title in zip(
    [data_all, data_inexperienced, data_experienced],
    ["All Participants", "Inexperienced Participants", "Experienced Participants"]
):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        data,
        annot=True,
        mask=mask,
        fmt="",
        annot_kws={"size": 9},
        cmap=cmap,
        norm=LogNorm(vmin=data[data > 0].min(), vmax=data.max()),
        cbar=False,
        square=True,
        ax=ax
    )

    for text in ax.texts:
        text.set_fontsize(font_size)

        val = float(text.get_text())
        if val < 0.001:
            new_text = r"$\mathbf{< 10^{-3}}$"
            # new_text = f"\\textbf{{{val:.3f}}}"
        elif val < 0.05:
            new_text = f"\\textbf{{{val:.1e}}}"
        else:
            new_text = sci_fmt(val)

        text.set_text(new_text)
        text.set_color('black')

    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0, va='center')

    # Style adjustments
    ax.tick_params(left=False, bottom=False, labelsize=font_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(
        f"/home/fryderyk/Documents/code/study-evaluation/outputs/heatmap_{title.replace(' ', '_').lower()}.jpeg", dpi=300)
    plt.close()

x = 0
