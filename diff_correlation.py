import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

import utils


def main() -> None:

    df = pd.read_csv('outputs/results.csv')
    df = utils.remove_calibration(df)

    # keep only those columns

    df = df[df["transform_type"] !=
            "TransformType.NONE"].reset_index(drop=True)

    df = df[df["task_id"] ==
            "recurrence"].reset_index(drop=True)
    df_copy = df.copy()

    user_ids = list(df["user_id"].unique())
    n_users = len(user_ids)

    fig, axes = plt.subplots(
        1,
        n_users,
        figsize=(5 * n_users, 5),
        sharey=True
    )

    if n_users == 1:
        axes = [axes]

    for ax, user_id in zip(axes, user_ids):
        df = df_copy.copy()
        df = df[df["user_id"] ==
                user_id].reset_index(drop=True)

        # df = df[df["recurrence_confusion"].isin(
        #     ['fp', 'tp'])].reset_index(drop=True)

        df = df[df["synchronised_duration_seconds"]
                != 0.0].reset_index(drop=True)

        # for each user_id transform_type combination calculate the respetive average duration_seconds
        avg_durations = df.groupby(['user_id', 'transform_type'])[
            'duration_seconds'].mean()

        # divide each duration_seconds by the respective average duration_seconds
        df['normalized_duration'] = df.apply(
            lambda row: row['duration_seconds'] /
            avg_durations.loc[(row['user_id'], row['transform_type'])],
            axis=1
        )

        keep = [
            'user_id',
            'patient_id',
            'transform_type',
            'task_id',
            'task_index',
            'duration_seconds',
            'recurrence_confusion',
            'recurrence_abs',
            'dsc',
            'normalized_duration'
        ]
        df = df[keep]

        cols = list(df.columns)
        cols.insert(cols.index("duration_seconds")+1,
                    cols.pop(cols.index("normalized_duration")))
        df = df[cols]

        """
        # keep only patients with both transform types - and remove duplicate transforms
        patients_with_transforms = df.groupby("patient_id")["transform_type"].apply(
            lambda x: {"TransformType.NONLINEAR",
                    "TransformType.LINEAR"}.issubset(set(x))
        ).to_dict()

        patients_with_both = [
            patient_id for patient_id, has_both in patients_with_transforms.items() if has_both
        ]

        df = df[df["patient_id"].isin(patients_with_both)].reset_index(drop=True)

        df = (
            df.groupby(["patient_id", "transform_type"])
            .head(1)  # keep first occurrence of each type
            .reset_index(drop=True)
        )

        # Pivot the table to have separate columns for LINEAR and NONLINEAR
        df = df.pivot(
            index="patient_id",
            columns="transform_type",
            values=["normalized_duration", "dsc"]
        )

        df = pd.DataFrame({
            "duration_diff": df["normalized_duration"]["TransformType.NONLINEAR"]
            - df["normalized_duration"]["TransformType.LINEAR"],
            "dsc_diff": df["dsc"]["TransformType.NONLINEAR"]
            - df["dsc"]["TransformType.LINEAR"]
        }).reset_index()
        """

        df.to_csv('temp.csv')

        pearson_r, pearson_p = pearsonr(
            df["dsc"], df["duration_seconds"])
        spearman_r, spearman_p = spearmanr(
            df["dsc"], df["duration_seconds"])

        print(f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.3e})")
        print(f"Spearman r = {spearman_r:.3f} (p = {spearman_p:.3e})")

        X = sm.add_constant(df["dsc"])
        y = df["duration_seconds"]

        model = sm.OLS(y, X).fit()
        print(model.summary())

        ax.scatter(df["dsc"], df["duration_seconds"], label="Patients")
        x_vals = np.linspace(df["dsc"].min(), df["dsc"].max(), 100)
        ax.plot(x_vals, model.params[0] + model.params[1]
                * x_vals, color="red", label="Regression line")
        ax.set_xlabel("DSC")
        ax.set_title(
            f"{user_id}: r={pearson_r:.3f} (p={pearson_p:.3e})")
        if ax is axes[0]:
            ax.set_ylabel("Normalized Duration")
        ax.legend()

    fig.tight_layout()
    plt.show()

    x = 0


if __name__ == "__main__":
    main()
