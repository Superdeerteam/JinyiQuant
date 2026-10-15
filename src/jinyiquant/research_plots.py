"""Optional research figures; core training and evaluation do not require matplotlib."""

from pathlib import Path


def plot_learning_curve(rows, path, model):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax in axes:
        for field, label in (("train_mse", "Training"), ("validation_mse", "Research validation")):
            points = [(row["epoch"], row[field]) for row in rows if row.get(field) is not None]
            if points:
                x, y = zip(*points)
                ax.plot(x, y, marker="o", label=label)
        ax.set(xlabel="Epoch / analytic checkpoint", ylabel="MSE (raw official return units squared)")
        ax.grid(alpha=.25)
    if any(row["train_mse"] > 0 for row in rows):
        axes[1].set_yscale("log")
    axes[0].legend()
    fig.suptitle(f"{model}: fixed weights at each checkpoint, no validation selection")
    fig.savefig(Path(path), dpi=140)
    plt.close(fig)


def plot_baseline_summary(variation, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    periods = list(dict.fromkeys(row["period"] for row in variation))
    models = ("mean", "ridge", "modified_tlob")
    colors = ("#78828c", "#00877a", "#bc4c42")
    fig, axes = plt.subplots(len(periods), 4, figsize=(15, 4 * len(periods)), squeeze=False,
                             constrained_layout=True)
    for row_index, period in enumerate(periods):
        rows = {row["model"]: row for row in variation if row["period"] == period}
        for column, (metric, title) in enumerate((("mse", "Validation MSE"), ("rank_ic_mean", "RankIC"),
                                                 ("long_short_return_mean", "High minus low return"))):
            ax = axes[row_index, column]
            for index, model in enumerate(models):
                value = rows[model]["metrics"][metric]
                if value["mean"] is None:
                    ax.text(index, .05, "undefined", ha="center", transform=ax.get_xaxis_transform(), fontsize=8)
                else:
                    ax.bar(index, value["mean"], color=colors[index], yerr=value["seed_std"] or 0, capsize=3)
            if metric == "mse":
                ax.set_yscale("log")
            ax.set(title=f"{period}: {title}", xticks=range(3), xticklabels=models)
            ax.grid(axis="y", alpha=.2)
        ax = axes[row_index, 3]
        for index, model in enumerate(models):
            seconds = [row["resources"]["wall_seconds"] for row in summary
                       if row["period"] == period and row["model"] == model]
            ax.bar(index, np.mean(seconds) / 60, color=colors[index])
        ax.set(title=f"{period}: full run minutes", xticks=range(3), xticklabels=models)
    fig.suptitle("Forward-date research baselines | 3 seeds | error bars: seed SD | local approximate metrics")
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)
