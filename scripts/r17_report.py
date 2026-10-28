#!/usr/bin/env python
"""R17 headline report: one JSON, no markdown (repo rule 13)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs" / "r17-20261029"
R6 = REPO / "runs" / "r6r8-20261029" / "screen"


def main() -> None:
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    model = json.loads((RUN / "model.json").read_text(encoding="utf-8"))
    predict_meta = json.loads((RUN / "predict_meta.json").read_text(encoding="utf-8"))
    train_total = sum(model["train_seconds"].values())
    predict_total = sum(v["seconds"] for v in predict_meta.values())
    days = 165
    r6_tree = json.loads((R6 / "summary.json").read_text(encoding="utf-8"))["pooled"]["tree"]["rank_ic"]
    fusion = summary["fusion"]
    primary = fusion["0.20"]
    chosen = fusion[f"{summary['chosen_weight_by_frozen']:.2f}"]
    verdict = ("fusion increment is positive and CI excludes zero on the honest later-six block"
               if primary["later"]["ci_low"] > 0 else
               "no reliable fusion increment: later-six CI includes zero")
    report = {
        "schema": "r17-report/1",
        "route": "R17 fixed MiniRocket-style conv features + ridge on the 128x12 raw window",
        "hypothesis": summary["hypothesis"],
        "guardrails": summary["guardrails"],
        "config": {
            "kernels": "21 of the 84 C(9,3) MiniRocket kernels (every 4th combination, fixed a priori)",
            "dilations": summary["model"]["dilations"],
            "features": summary["model"]["features"],
            "bias": "one golden-ratio quantile per (kernel,dilation), pooled over 8192 training windows (2023-10)",
            "readout": f"ridge lambda={summary['model']['lambda']} on per-timestamp CSZ(vwap_return) clip +-5",
            "train_rows": summary["model"]["rows"],
            "train_months": "2022-01..2023-10 (same era as the 441-day MLPLOB backbone)",
            "input": "canonical 128x12 window, configs/441_lotb_replay_plan.json data block, ffill<=5",
            "deviations": summary["deviations"],
        },
        "single_model": {
            "pooled": summary["pooled"]["rocket"],
            "per_window_rank_ic": summary["per_window"]["rocket"],
            "baseline_r441": summary["pooled"]["blend_r441"],
            "baseline_r441p": summary["pooled"]["blend_r441p"],
            "r6_tree_pooled_rank_ic": r6_tree,
        },
        "fusion": {
            "protocol": "(1-w)*z(blend_r441) + w*z(rocket), per timestamp; w frozen on nov2023+jan2024, "
                        "honest read on feb/mar/apr/may/jul/aug (121 days)",
            "weights": summary["weights"],
            "chosen_weight_by_frozen": summary["chosen_weight_by_frozen"],
            "primary_w0.20": {"frozen": primary["frozen"], "later": primary["later"], "all": primary["all"],
                              "vs_r441p_all": primary["vs_r441p_all"]},
            "chosen": {"w": summary["chosen_weight_by_frozen"], "later": chosen["later"], "all": chosen["all"]},
            "pooled_fused": {k: v["rank_ic"] for k, v in summary["pooled"].items() if k.startswith("fused")},
        },
        "diagnostics": {
            "correlation_with_blend": summary["diagnostics"]["correlation"],
            "residual_ic_vs_blend": summary["diagnostics"]["residual_ic"],
            "rank_correlation_with_r6_tree": summary["diagnostics"].get("rocket_vs_r6"),
        },
        "cost": {
            "fit_seconds_total": round(train_total, 1),
            "fit_seconds_per_month": model["train_seconds"],
            "bias_fit_seconds": 14.9,
            "predict_seconds_total": round(predict_total, 1),
            "predict_seconds_per_window": {k: v["seconds"] for k, v in predict_meta.items()},
            "marginal_seconds_per_trading_day": round((train_total + predict_total) / days, 2),
            "note": "local CPU (12 cores shared with replay jobs), numpy single-thread; ridge readout is pure numpy",
        },
        "verdict": verdict,
        "comparison_r6": {
            "r6_tree_pooled_rank_ic": r6_tree,
            "r6_fusion_delta_w0.20_all": 0.00567,
            "r17_fusion_delta_w0.20_all": primary["all"]["mean_delta"],
            "r17_single_vs_r6_tree": summary["pooled"]["rocket"]["rank_ic"] - r6_tree,
        },
        "artifacts": {
            "script": "scripts/r17_minirocket.py",
            "predictions": "runs/r17-20261029/predictions/<window>.parquet (date, instrument, score)",
            "summary": "runs/r17-20261029/summary.json",
            "model": "runs/r17-20261029/model.json",
            "daily": "runs/r17-20261029/daily.parquet",
            "fit_log": "runs/r17-20261029/fit.log",
            "predict_log": "runs/r17-20261029/predict.log",
        },
        "not_run_by_design": "captain's R6+R8 round-2 follow-ups (3 seeds, gain-pruned feature subsets, 927-day "
                             "training, train-vs-post-2023-11 IC gap) wait for the captain's registration",
    }
    (RUN / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verdict",)}, indent=1))
    print(json.dumps(report["fusion"]["primary_w0.20"]["all"], indent=1)[:600])


if __name__ == "__main__":
    main()
