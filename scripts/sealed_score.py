"""Score the one-shot sealed window for the frozen candidate and the reference blend."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rung_candidate_compare as rc  # noqa: E402

INCENT = {"members": [(None, f"inc-{kind}-s{seed}") for kind in ("tlob", "mlp", "patch")
                      for seed in (11, 29, 47)]}
CAND = {"members": [(None, f"cand-s{seed}-264") for seed in (11, 29, 47)]}


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2])
    rc.WINDOWS = ["sealed2024q4"]
    for spec in (INCENT, CAND):
        spec["members"] = [(str(run_dir), version) for _, version in spec["members"]]

    reference = rc.run_candidate("reference-nine", INCENT)
    candidate = rc.run_candidate("candidate-441-264", CAND)

    left = reference["daily"].rename(columns={"rank_ic": "rank_ic_ref"})
    right = candidate["daily"].rename(columns={"rank_ic": "rank_ic_cand"})
    paired = left[["date", "rank_ic_ref"]].merge(right[["date", "rank_ic_cand"]],
                                                 on="date", validate="one_to_one")
    paired["delta"] = paired["rank_ic_cand"] - paired["rank_ic_ref"]
    delta = paired["delta"].to_numpy(dtype=float)
    interval = rc.block_interval(delta, block=5, draws=20000)

    report = {
        "window": "sealed2024q4",
        "start": "2024-10-08",
        "end": "2025-01-01",
        "days": int(len(delta)),
        "candidate": {
            "alpha": rc.ALPHA,
            "rank_ic_smoothed": candidate["rank_ic_smoothed"],
            "rank_ic_unsmoothed": candidate["rank_ic_unsmoothed"],
            "turnover_smoothed": candidate["turnover_smoothed"],
            "long_short_mean": candidate["long_short_mean"],
        },
        "reference_nine_blend": {
            "rank_ic": reference["rank_ic_unsmoothed"],
            "turnover": reference["turnover_smoothed"] - 0.0,
            "long_short_mean": reference["long_short_mean"],
        },
        "paired_candidate_minus_reference": {
            "mean_delta": float(delta.mean()),
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "positive_days": int((delta > 0).sum()),
        },
        "paired_daily": paired.assign(window="sealed2024q4").to_dict(orient="records"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_daily"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
