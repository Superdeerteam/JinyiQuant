"""Plot fixed archived research comparisons; no training or model selection."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PERIODS = {"july_18_21": "July 18-21, 2023", "july_25_28": "July 25-28, 2023"}
SEEDS = (11, 29, 47)
COLORS = ("#2468A0", "#198576", "#9A5B30")
METRICS = (("rank_ic", "rank_ic_mean", "RankIC", 1, "Higher is better"),
           ("long_short_return", "long_short_return_mean", "Gross long-short return (bp)", 10000,
            "Higher is better; before costs"),
           ("turnover", "turnover_mean", "Turnover", 1, "Lower is better"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(report):
    rows = {(r["period"], r["seed"], r["arm"]): r for r in report["summary"]}
    expected = {(p, s, a) for p in PERIODS for s in SEEDS for a in ("baseline", "candidate")}
    if not report["checks"]["passed"] or set(rows) != expected or len(report["summary"]) != len(expected):
        raise ValueError("Figure requires the intact two-period, three-seed comparison")
    return rows


def style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CDD5DE")
    ax.tick_params(axis="both", length=0, labelcolor="#42536A")
    ax.set_axisbelow(True)


def plot_times(report, decision, output, source_sha):
    rows = grid(report)
    intervals = {(r["period"], r["metric"]): r for r in report["paired_intervals"]}
    assert all(r["steps"] == 1500 for r in rows.values())
    assert decision["comparison_sha256"] == source_sha
    opening = decision["opening_changes"]
    worse = sum(r["rank_ic_delta"] < 0 and r["long_short_delta_bp"] < 0 for r in opening)
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 9.4), sharex="col")
    fig.subplots_adjust(left=.105, right=.97, top=.82, bottom=.34, hspace=.50, wspace=.30)
    fig.suptitle("More intraday training times: paired research results", x=.06, y=.965,
                 ha="left", fontsize=21, fontweight="bold", color="#162F49")
    fig.text(.06, .918, "2 to 8 training times | 10 training dates | 2 validation windows x 3 seeds",
             fontsize=12, color="#52657A")
    plotted = []
    for col, (metric, field, title, scale, direction) in enumerate(METRICS):
        all_values = [0.0]
        for row, (period, label) in enumerate(PERIODS.items()):
            ax = axes[row, col]
            style_axis(ax)
            ax.axvline(0, color="#7F8B98", linestyle="--", linewidth=1)
            ax.grid(axis="x", color="#E9EDF2", linewidth=.8)
            ax.axhspan(-.38, .38, color="#EFF4F8", zorder=0)
            ax.set_ylim(-.55, 3.6)
            ax.set_yticks([3, 2, 1, 0], ["Seed 11", "Seed 29", "Seed 47", "Day-paired mean"])
            if col:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title(f"{title}\n{label}", loc="left", fontsize=11, color="#162F49", pad=13)
            for index, seed in enumerate(SEEDS):
                baseline, candidate = (rows[period, seed, arm][field] * scale
                                       for arm in ("baseline", "candidate"))
                delta = candidate - baseline
                ax.plot(delta, 3 - index, "o", color=COLORS[index], markersize=7)
                ax.annotate(f"{delta:+.3f}" if scale == 1 else f"{delta:+.2f}",
                            (delta, 3 - index), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=10, color="#263C52")
                all_values.append(delta)
                plotted.append({"period": period, "seed": seed, "metric": metric,
                                "baseline": baseline, "candidate": candidate, "delta": delta,
                                "scale": scale})
            interval = intervals[period, metric]
            assert interval["status"] == "estimated" and interval["days"] == 4 and interval["seeds"] == 3
            mean = interval["mean"] * scale
            low, high = np.asarray(interval["interval"]) * scale
            ax.hlines(0, low, high, color="#263C52", linewidth=3)
            ax.plot([low, high], [0, 0], "|", color="#263C52", markersize=10)
            ax.plot(mean, 0, "D", color="#263C52", markersize=6)
            all_values.extend((low, high, mean))
            if row == 1:
                ax.set_xlabel(f"8-time minus 2-time training\n{direction}", fontsize=10, labelpad=11)
        lo, hi = min(all_values), max(all_values)
        span = hi - lo
        axes[0, col].set_xlim(lo - .10 * span, hi + .29 * span)
        axes[0, col].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.text(.06, .214,
             f"Opening risk: 09:30 RankIC and gross long-short decline in {worse}/{len(opening)} pairs.",
             fontsize=12, color="#883A26", fontweight="bold",
             bbox={"facecolor": "#FFF1E8", "edgecolor": "none", "pad": 9})
    epochs = {arm: np.mean([r["epochs"] for k, r in rows.items() if k[2] == arm])
              for arm in ("baseline", "candidate")}
    notes = [
        f"Both arms evaluate all 8 times and use 1,500 optimizer updates per run (batch 32). "
        f"Pool passes: 2-time {epochs['baseline']:.2f}; 8-time {epochs['candidate']:.2f}.",
        "Diamonds: paired mean; bars: archived 95% moving 2-day-block intervals, averaging seeds within each day.",
        "Only 8 research dates (4 per window); prior screening and date dependence limit generalization. "
        "Local metrics, not an official competition score.",
        f"Source: step3c-full-analysis/comparison.json | SHA256 {source_sha[:16]}... | full provenance in adjacent JSON.",
    ]
    for y, note in zip((.157, .122, .087, .047), notes):
        fig.text(.06, y, note, fontsize=9.3, color="#52657A")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return {"values": plotted, "intervals": [intervals[p, m] for p in PERIODS for m, *_ in METRICS],
            "opening_pairs_worse": worse, "opening_pairs": len(opening), "epochs": epochs}


def plot_epochs(report, output, source_sha):
    rows = grid(report)
    assert report["plan"]["change"]["factor"] == "model.epochs"
    assert [report["plan"]["change"][a] for a in ("baseline", "candidate")] == [2, 12]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 6.1), sharey=True)
    fig.subplots_adjust(left=.12, right=.96, top=.73, bottom=.27, wspace=.22)
    fig.suptitle("Separate experiment: 2 versus 12 training epochs", x=.07, y=.96,
                 ha="left", fontsize=18, fontweight="bold", color="#162F49")
    fig.text(.07, .895, "Validation MSE falls in each paired seed; lower error alone does not establish alpha.",
             fontsize=10.5, color="#52657A")
    plotted = []
    for ax, (period, label) in zip(axes, PERIODS.items()):
        style_axis(ax)
        for color, seed in zip(COLORS, SEEDS):
            values = [rows[period, seed, arm]["validation_mse"] for arm in ("baseline", "candidate")]
            ax.plot([0, 1], values, "o-", label=f"Seed {seed}", color=color, linewidth=1.8)
            plotted.append({"period": period, "seed": seed, "epochs": [2, 12], "validation_mse": values})
        ax.set_title(label, fontsize=12, color="#162F49", pad=12)
        ax.set_xticks([0, 1], ["2 epochs", "12 epochs"])
        ax.set_xlim(-.15, 1.15)
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", color="#E9EDF2")
    axes[0].set_ylabel("Validation MSE (return squared; logarithmic scale)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")
    fig.text(.07, .17, "Fixed data, inputs and Adam learning rate (1e-4) within each pair. "
             "This is an independent control experiment.", fontsize=9.5, color="#52657A")
    fig.text(.07, .115, "Same eight research dates; local validation only, not an official score. "
             "No final holdout was opened for these figures.", fontsize=9.5, color="#52657A")
    fig.text(.07, .055, f"Source: step1-analysis/comparison.json | SHA256 {source_sha[:16]}...",
             fontsize=9, color="#52657A")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return {"values": plotted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "runs/training-progression")
    args = parser.parse_args()
    root = args.results.resolve()
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {"times": root / "step3c-full-analysis/comparison.json",
               "decision": root / "step3c-full-decision.json", "epochs": root / "step1-analysis/comparison.json"}
    reports = {key: read(path) for key, path in sources.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    outputs = {"times": destination / "step3c_training_times_paired.png",
               "epochs": destination / "step1_epochs_validation_mse.png"}
    evidence = {"times": plot_times(reports["times"], reports["decision"], outputs["times"], sha(sources["times"])),
                "epochs": plot_epochs(reports["epochs"], outputs["epochs"], sha(sources["epochs"]))}
    metadata = {"schema_version": 1, "script": str(Path(__file__).resolve()), "script_sha256": sha(Path(__file__)),
                "sources": {key: {"path": str(path), "sha256": sha(path)} for key, path in sources.items()},
                "figures": {key: {"path": str(path), "sha256": sha(path), **evidence[key]}
                            for key, path in outputs.items()},
                "training_started": False, "model_promoted": False, "final_views": 0}
    (destination / "training_progression_figures.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: str(path) for key, path in outputs.items()}))


if __name__ == "__main__":
    main()
