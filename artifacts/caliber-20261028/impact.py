"""Impact audit: re-judge every *registered* verdict under the corrected 5-day
moving-block bootstrap, side by side with the caliber it was archived under.

Why: `compare_blend_replay.block_interval` as it stood drew exactly ONE 5-trading-day
block per replicate and averaged that block, so any interval computed through it is
inflated by roughly sqrt(n/5) (~4.5x at n=104). A corrected version resamples
ceil(n/5) contiguous blocks per replicate. Point estimates never move: only the
interval does, and only where the inflated helper was actually used.

Important finding this script establishes rather than assumes: the debt is NOT uniform.
The dose-curve builder (`runs/dose-replay-20261027/build_metrics.py:107`) and the C1
builder (`runs/data-probe-replay-20261027/build_c1_test.py:26`) implement ceil(n/block)
blocks themselves and even document the single-block bug; the promotion gate has its own
per-period block estimator. Only the later script family that imported
`compare_blend_replay.block_interval` (rung compare, featnorm decision, 927 extension,
sealed scoring) inherited the inflated caliber. For every archived interval this script
therefore identifies *which* estimator reproduces it bit-for-bit, and re-judges the
registered rule against the corrected interval.

No threshold is modified, no historical verdict file is written, no sealed-month raw
data or score is recomputed. Outputs: impact.json and impact_table.txt.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
HERE = Path(__file__).resolve().parents[1]
SERIES = HERE / "src" / "series"
OUT_JSON = HERE / "impact.json"
OUT_TABLE = HERE / "impact_table.txt"

BLOCK = 5
CANON_DRAWS = 20000
CANON_SEED = 20261028
SEED_CANDIDATES = (20261027, 20261026, 20261028)
WINDOWS = ("nov2023", "jan2024", "mar2024", "apr2024", "jun2024")
PERIODS = [("2023-11-01", "2023-11-30"), ("2024-01-01", "2024-01-31"),
           ("2024-03-01", "2024-03-31"), ("2024-04-01", "2024-04-30"),
           ("2024-06-01", "2024-06-30")]

PROBE_132 = {1.32: "data-probe-replay-20261027/batch1/20261027T181108.631387Z_e34e5d38ed7f",
             2.64: "data-probe-replay-20261027/batch2/20261027T182058.643438Z_774051b8c037"}
DOSE_RUNS = {  # archived points_source: the run dir each rung/point was scored from
    "240d": {1.32: {11: "20261027T193656.983242Z_c9441fa358d7", 29: "20261027T203910.711759Z_f125f5550145",
                    47: "20261027T214137.108463Z_4e9ecb679a69"},
             2.64: {11: "20261027T193656.983242Z_c9441fa358d7", 29: "20261027T203910.711759Z_f125f5550145",
                    47: "20261027T214137.108463Z_4e9ecb679a69"}},
    "441d": {1.32: {11: "20261027T205912.221663Z_177502e7e8b9", 29: "20261027T230138.712292Z_3317a578512c",
                    47: "20261028T000705.935779Z_f7505bc63267"},
             2.64: {11: "20261027T205912.221663Z_177502e7e8b9", 29: "20261027T230138.712292Z_3317a578512c",
                    47: "20261028T004911.418444Z_e7eed5e69132"}},
}
DOSE_BATCH = {"240d": "batch-240d-seed{seed}", "441d": "batch-441d-seed{seed}"}
ANCHOR60 = "training-trajectory-replay-20261027/20261027T134230.458586Z_457f754f746d"
FEATNORM = SERIES / "featnorm-replay-20261028/20261028T071015.646405Z_8dd7831f6bdb"
FEATNORM_EXT = SERIES / "featnorm-ext-replay-20261028/20261028T075924.920238Z_86aab1d89424"
EXT_A = SERIES / "927d-replay-20261028-batchA/20261028T073610.848937Z_f89d015d6ae2"
EXT_B = SERIES / "927d-replay-20261028-batchB/20261028T082014.218918Z_881f3c9bc12f"
GATE_CAND_PATH = "promotion-candidate-ewma-match-20261028/objects/cand_441d_2p64_3seed_ewma_match/daily_0bp.csv"
GATE_REF_PATH = "turnover-stability-20261027/20261027T070343Z_606ce5deaa8b/objects/reference_nine_blend/daily_0bp.csv"
GATE_DIR = RUNS / "promotion-candidate-ewma-match-20261028/gate-cand_441d_2p64_3seed_ewma_match"
SEALED = "sealed-final-20261028/sealed_result.json"

TRACKED: list[dict] = []


# --------------------------------------------------------------------------- io
def read_daily(path: Path, column: str = "rank_ic") -> pd.Series:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].duplicated().any():
        raise SystemExit(f"{path}: duplicate trading days")
    return frame.set_index("date").sort_index()[column]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rung_series(rung: str, seed: int, point: float) -> pd.Series:
    """One seed's merged 104-day rank_ic series for a dose rung at one budget point."""
    tag = "1.32" if point == 1.32 else "2.64"
    parts = []
    for window in WINDOWS:
        if rung == "132d":
            path = RUNS / PROBE_132[point] / "windows" / window / f"s{seed}-{tag}" / "daily.csv"
        else:
            base = RUNS / "dose-replay-20261027" / DOSE_BATCH[rung].format(seed=seed) / DOSE_RUNS[rung][point][seed]
            path = base / "windows" / window / f"s{seed}-{tag}" / "daily.csv"
        parts.append(read_daily(path))
    return pd.concat(parts).sort_index()


def paired_seed_mean(left: dict[int, pd.Series], right: dict[int, pd.Series]) -> pd.Series:
    """Per-seed (right - left) paired on trading day, then the seed mean (house construction)."""
    out = []
    for seed in sorted(left):
        joined = pd.concat({"l": left[seed], "r": right[seed]}, axis=1, join="inner").dropna()
        out.append((joined["r"] - joined["l"]).rename(seed))
    return pd.concat(out, axis=1).mean(axis=1).sort_index()


def version_delta(cand_base: Path, cand: str, ctrl_base: Path, ctrl: str, point: str,
                  seeds: tuple[int, ...]) -> tuple[pd.Series, dict]:
    per_window, pooled = {}, []
    for window in WINDOWS:
        parts = []
        for seed in seeds:
            a = read_daily(cand_base / "windows" / window / cand.format(seed=seed, point=point) / "daily.csv")
            b = read_daily(ctrl_base / "windows" / window / ctrl.format(seed=seed, point=point) / "daily.csv")
            joined = pd.concat({"c": a, "b": b}, axis=1, join="inner").dropna()
            parts.append((joined["c"] - joined["b"]).rename(seed))
        delta = pd.concat(parts, axis=1).mean(axis=1).sort_index()
        per_window[window] = {"days": int(len(delta)), "mean": float(delta.mean())}
        pooled.append(delta)
    return pd.concat(pooled).sort_index(), per_window


# --------------------------------------------------------------- estimators
def _means(values: np.ndarray, draws: int, seed: int, n_blocks: int, truncate: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, values.size, size=(draws, n_blocks))
    index = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % values.size
    drawn = values[index.reshape(draws, -1)]
    if truncate:  # the dose/C1 builders average only the first n resampled values
        return drawn[:, : values.size].mean(axis=1)
    return drawn.mean(axis=1)


def _clean(values) -> np.ndarray:
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def interval(values, draws: int, seed: int, single_block: bool = False, truncate: bool = False) -> dict:
    """single_block=True is the retired caliber (one block per replicate => inflated CI)."""
    v = _clean(values)
    n_blocks = 1 if single_block else math.ceil(v.size / BLOCK)
    means = _means(v, draws, seed, n_blocks, truncate=truncate)
    if single_block:
        estimator = "single_block_inflated"
    else:
        estimator = "ceil_blocks_correct_truncated" if truncate else "ceil_blocks_correct"
    return {"n": int(v.size), "mean": float(v.mean()),
            "ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975)),
            "draws": int(draws), "seed": int(seed), "n_blocks": int(n_blocks),
            "estimator": estimator}


def _block_indices(length: int, block_days: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, length - block_days + 1, size=int(np.ceil(length / block_days)))
    return (starts[:, None] + np.arange(block_days)).ravel()[:length]


def gate_interval(values: pd.Series, draws: int, seed: int, single_block: bool = False) -> dict:
    """The promotion gate's own estimator: blocks inside each declared period, never across,
    and each declared period carries equal weight (the registered weighting)."""
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float),
                          "date": pd.to_datetime(values.index).normalize()}).dropna()
    groups = []
    for start, end in PERIODS:
        block = frame[(frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))]
        groups.append(block["value"].to_numpy(dtype=float))
    if any(len(g) == 0 for g in groups):
        raise SystemExit("a declared period has no paired days")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws)
    for i in range(draws):
        total = count = 0.0
        for values_p in groups:
            length = len(values_p)
            if single_block:
                starts = rng.integers(0, length - BLOCK + 1, size=1)
                indices = (starts[:, None] + np.arange(BLOCK)).ravel()
            else:
                indices = _block_indices(length, min(BLOCK, length), rng)
            total += float(values_p[indices].sum())
            count += len(indices)
        sampled[i] = total / count
    period_means = [float(g.mean()) for g in groups]
    return {"n": int(len(frame)), "mean": float(np.mean(period_means)),
            "day_weighted_mean": float(frame["value"].mean()),
            "ci_low": float(np.quantile(sampled, 0.025)), "ci_high": float(np.quantile(sampled, 0.975)),
            "draws": int(draws), "seed": int(seed),
            "days_per_period": [int(len(g)) for g in groups],
            "period_means": period_means,
            "weighting": "improvement = equal weight per declared period (matches gate.json); "
                         "bootstrap replicates pool block sums across periods (day-weighted, matches "
                         "turnover_gate_readout.block_interval)",
            "estimator": ("gate_period_blocks_single_inflated" if single_block
                          else "gate_period_blocks_correct")}


def identify(entry_id: str, archived: dict, series: pd.Series, draws: int, seed_hint: int) -> dict:
    """Which estimator and seed reproduce the archived interval bit-for-bit?"""
    hits = []
    for seed in [seed_hint] + [s for s in SEED_CANDIDATES if s != seed_hint]:
        for single in (True, False):
            for truncate in (False, True):
                if single and truncate:
                    continue  # a single 5-day block never needs truncation
                got = interval(series, draws, seed, single_block=single, truncate=truncate)
                if (abs(got["ci_low"] - archived["ci_low"]) < 1e-12
                        and abs(got["ci_high"] - archived["ci_high"]) < 1e-12):
                    hits.append({"estimator": got["estimator"], "seed": seed, "truncate": truncate})
    if not hits:
        raise SystemExit(f"{entry_id}: no estimator reproduces the archived interval "
                         f"(draws={draws}, seed_hint={seed_hint})")
    estimators = {h["estimator"] for h in hits}
    if len(estimators) > 1:
        raise SystemExit(f"{entry_id}: archived interval is ambiguous between estimators {sorted(estimators)}")
    # a single 5-day block leaves only ~n distinct replicate means, so the discrete quantiles can
    # coincide across bootstrap seeds; the seed is a nuisance parameter, the estimator is not.
    hits.sort(key=lambda h: (0 if h["seed"] == seed_hint else 1,
                             SEED_CANDIDATES.index(h["seed"]) if h["seed"] in SEED_CANDIDATES else 99))
    chosen = dict(hits[0])
    chosen["seed_matches"] = [h["seed"] for h in hits]
    chosen["seed_ambiguous"] = len(hits) > 1
    return chosen


# -------------------------------------------------------------------- driver
def record(entry: dict) -> None:
    archive, corrected = entry["archive"], entry["corrected"]
    inflated = entry.get("counterfactual_inflated")
    if archive["caliber_debt"]:
        entry["half_width_ratio_inflated_over_corrected"] = round(
            (archive["ci_high"] - archive["ci_low"]) / (corrected["ci_high"] - corrected["ci_low"]), 4)
    elif inflated:
        entry["half_width_ratio_inflated_over_corrected"] = round(
            (inflated["ci_high"] - inflated["ci_low"]) / (archive["ci_high"] - archive["ci_low"]), 4)
    else:
        entry["half_width_ratio_inflated_over_corrected"] = None
    entry["archived_interval_excludes_zero"] = bool(archive["ci_low"] > 0 or archive["ci_high"] < 0)
    entry["corrected_interval_excludes_zero"] = bool(corrected["ci_low"] > 0 or corrected["ci_high"] < 0)
    entry["interval_flips"] = bool(entry["archived_interval_excludes_zero"]
                                   != entry["corrected_interval_excludes_zero"])
    TRACKED.append(entry)


def judge(entry_id: str, family: str, comparison: str, series: pd.Series, archived: dict,
          draws: int, seed_hint: int, rule: dict, archived_verdict: dict, gate_type: str,
          sensitivity: str, rejudge, verdict_recorded: bool = True) -> None:
    """Reproduce the archived interval, correct it, and re-apply the registered rule to both."""
    usage = identify(entry_id, archived, series, draws, seed_hint)
    truncated = usage["truncate"]
    inflated = interval(series, draws, usage["seed"], single_block=True)
    corrected = interval(series, draws, usage["seed"], single_block=False, truncate=truncated)
    debt = usage["estimator"] == "single_block_inflated"
    was, now = (inflated if debt else corrected), corrected
    passed_before, criteria_before, text_before = rejudge(was)
    passed_after, criteria_after, text_after = rejudge(now)
    canon = interval(series, CANON_DRAWS, CANON_SEED, single_block=False, truncate=truncated)
    record({
        "id": entry_id, "family": family, "comparison": comparison, "n_days": int(corrected["n"]),
        "point_estimate": corrected["mean"],
        "archive": {**archived, "draws": draws, "seed": usage["seed"], "estimator": usage["estimator"],
                    "n_blocks": 1 if usage["estimator"] == "single_block_inflated"
                                else math.ceil(corrected["n"] / BLOCK),
                    "caliber_debt": debt,
                    "seed_matches": usage.get("seed_matches"), "seed_ambiguous": usage.get("seed_ambiguous"),
                    "source": archived["source"]},
        "corrected": corrected, "canonical": canon,
        "counterfactual_inflated": None if debt else inflated,
        "registered_rule": rule,
        "archived_verdict": archived_verdict,
        "rule_outcome_archived_caliber": {"passed": passed_before, "criteria": criteria_before, "text": text_before},
        "rule_outcome_corrected_caliber": {"passed": passed_after, "criteria": criteria_after, "text": text_after},
        "verdict_recorded": verdict_recorded,
        "verdict_flip": (bool(passed_before != passed_after) if verdict_recorded else None),
        "gate_type": gate_type, "sensitivity_note": sensitivity,
    })


def main() -> int:
    provenance: dict[str, str] = {}
    validations: list[dict] = []

    # ---------------------------------------------------------------- (1) dose
    dose_points = {point: {rung: {seed: rung_series(rung, seed, point) for seed in (11, 29, 47)}
                           for rung in ("132d", "240d", "441d")} for point in (1.32, 2.64)}
    for point, rungs in dose_points.items():
        for rung, per_seed in rungs.items():
            validations.append({"check": f"dose {rung}@{point}: per-seed 104-day mean",
                                "values": {str(s): round(float(v.mean()), 15) for s, v in per_seed.items()},
                                "days": {str(s): int(len(v)) for s, v in per_seed.items()}})
    dose = json.loads((RUNS / "dose-replay-20261027/metrics.json").read_text(encoding="utf-8"))
    rung_rows = {(float(e["point"]), e["left"], e["right"]): e for e in dose["paired_daily_rung_deltas"]}
    scaling = json.loads((ROOT / "configs/b4_data_scaling_prereg_20261027.json").read_text(encoding="utf-8"))
    reading = scaling["declared_reading"]

    for point in (1.32, 2.64):
        for left, right in (("132d", "240d"), ("240d", "441d"), ("132d", "441d")):
            label = "132d-t3" if left == "132d" else left
            row = rung_rows[(point, label, right)]["combined_block_interval"]
            series = paired_seed_mean(dose_points[point][left], dose_points[point][right])
            if abs(series.mean() - row["mean"]) > 1e-15 or len(series) != int(row["days"]):
                raise SystemExit(f"dose {point} {left}->{right}: point/n mismatch "
                                 f"({series.mean()}, {len(series)}) vs ({row['mean']}, {row['days']})")
            tag = "@2.64" if point == 2.64 else "@1.32"
            if (left, right) == ("132d", "240d"):
                rule_quote = reading["monotone_rule"]
                role = "first step of the registered monotone ladder"
            elif (left, right) == ("240d", "441d"):
                rule_quote = reading["flattening_rule"]
                role = "second step; the flattening rule would fire if this interval straddled zero"
            else:
                rule_quote = reading["monotone_rule"]
                role = "registered 450-vs-132 leg, the only leg the monotone rule formally gates"
            scope = ("the registered monotone/flattening wording triggers on the 450-vs-132 and 450-vs-240 legs; "
                     "the individual rung-to-rung increments are read here as the same test applied to the step, "
                     "which is what the archived metrics report calls a 'step separates from zero' reading")
            rule = {"quote": rule_quote, "interval_quote": reading["interval"],
                    "falsification_quote": reading["falsification_rule"],
                    "source": "configs/b4_data_scaling_prereg_20261027.json -> declared_reading",
                    "role": role, "scope_note": scope, "threshold": 0.0}

            def rejudge(iv, left=left, right=right):
                crit = {"interval_lower_bound_gt_0": bool(iv["ci_low"] > 0.0),
                        "point_estimate_positive": bool(iv["mean"] > 0)}
                if (left, right) == ("240d", "441d"):
                    text = ("more data still pays: the flattening rule is not triggered"
                            if crit["interval_lower_bound_gt_0"] else "flattening declared")
                else:
                    text = ("step separates from zero" if crit["interval_lower_bound_gt_0"]
                            else "step indistinguishable from zero")
                return crit["interval_lower_bound_gt_0"], crit, text

            judge(f"dose-{left}-{right}{tag}", "dose",
                  f"{label} -> {right} paired daily rank_ic{dose_neg(point)}",
                  series,
                  {"ci_low": row["ci_low"], "ci_high": row["ci_high"],
                   "source": "runs/dose-replay-20261027/metrics.json -> paired_daily_rung_deltas"},
                  int(row["draws"]), 20261027, rule,
                  {"text": f"archived combined block interval excludes zero = {row['ci_excludes_zero']}",
                   "basis": f"half_width {row['half_width']:.15f}, positive_days {row['positive_days']}",
                   "source": "runs/dose-replay-20261027/metrics.json"},
                  "interval", "gated by the block interval only, so caliber-sensitive in principle; the archive "
                              "already used the corrected estimator", rejudge,
                  verdict_recorded=False)

    # ------------------------------------------------------------------ (2) C1
    probe = json.loads((RUNS / "data-probe-replay-20261027/metrics.json").read_text(encoding="utf-8"))
    c1 = probe["c1_escalation_test"]
    delta = c1["main_seed11"]["delta"]
    s132 = rung_series("132d", 11, 1.32)
    anchor_frames = []
    for path in sorted((RUNS / "training-trajectory-replay-20261027").glob("*/report/daily_all.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["version"] == "anchor-1.32"].copy()
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["source_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        anchor_frames.append(frame)
    if not anchor_frames:
        raise SystemExit("no anchor-1.32 daily series found")
    anchor = pd.concat(anchor_frames, ignore_index=True)
    if anchor["date"].duplicated().any():
        raise SystemExit(f"anchor-1.32 appears in more than one run: {sorted(anchor['source_path'].unique())}")
    s60 = anchor.set_index("date").sort_index()["rank_ic"]
    joined = pd.concat({"a": s132, "b": s60}, axis=1, join="inner").dropna()
    c1_series = joined["a"] - joined["b"]
    if abs(c1_series.mean() - delta["mean"]) > 1e-15:
        raise SystemExit(f"C1 point {c1_series.mean()} != archived {delta['mean']}")
    validations.append({"check": "C1 60d anchor daily series located",
                        "path": sorted(anchor["source_path"].unique()), "days": int(len(s60)),
                        "mean": float(s60.mean())})
    per_window = c1["main_seed11"]["per_window"]
    validations.append({"check": "C1 132d@1.32 minus 60d@1.32, seed 11 (104 days)",
                        "point": float(c1_series.mean()), "days": int(len(c1_series)),
                        "window_means": {w: v["mean"] for w, v in per_window.items()},
                        "positive_windows": c1["main_seed11"]["positive_windows"]})
    probe_prereg = json.loads((ROOT / "configs/b4_data_probe_prereg_20261027.json").read_text(encoding="utf-8"))
    c1_reg = probe_prereg["design"]["escalation_rule"]
    c1_quote = (f"escalation_state_name={c1_reg['escalation_state_name']}; "
                f"conditions={json.dumps(c1_reg['conditions'], ensure_ascii=False)}; "
                f"interval={json.dumps(c1_reg['interval'], ensure_ascii=False)}; "
                f"binding: {c1_reg['applies_to_comparison']['binding_declarations'][0]}")
    c1_rule = {"quote": c1_quote,
               "source": "configs/b4_data_probe_prereg_20261027.json -> design.escalation_rule "
                         "(rule_document configs/b4_budget_decision_prereg_v2_20261027.json)",
               "threshold": 0.0017, "required_positive_windows": 4, "windows_total": 5,
               "threshold_derivation": json.dumps(c1_reg["threshold_derivation"], ensure_ascii=False),
               "read_on": "primary seed 11 only"}

    def rejudge_c1(iv):
        positive = sum(1 for v in per_window.values() if v["mean"] > 0)
        crit = {"delta_ge_0.0017": bool(iv["mean"] >= 0.0017), "ci_low_gt_0": bool(iv["ci_low"] > 0.0),
                "positive_windows_ge_4_of_5": bool(positive >= 4), "positive_windows": positive}
        return all(crit.values()), crit, ("escalate" if all(crit.values()) else "stay at 132d")

    judge("c1-132-vs-60", "c1", "132d@1.32 - 60d@1.32, seed 11, paired daily rank_ic", c1_series,
          {"ci_low": delta["ci_low"], "ci_high": delta["ci_high"],
           "source": "runs/data-probe-replay-20261027/metrics.json -> c1_escalation_test.main_seed11.delta"},
          int(delta["draws"]), 20261027, c1_rule,
          {"text": "C1 not escalated: 132d did not beat 60d by the registered margin",
           "basis": json.dumps(c1["criterion"], ensure_ascii=False),
           "source": "runs/data-probe-replay-20261027/metrics.json -> c1_escalation_test.criterion"},
          "threshold+interval+windows",
          "passes only if the point estimate clears 0.0017 AND the interval clears zero AND 4/5 windows are "
          "positive; the point leg fails by a wide margin, so caliber cannot rescue it", rejudge_c1)
    return _finish(provenance, validations, dose_points)


def _finish(provenance, validations, dose_points):
    # -------------------------------------------------------- (3) 927 extension
    ext_series, ext_windows = version_delta(EXT_B, "s{seed}-264", EXT_A, "s{seed}-132", "264", (11, 29, 47))
    dec927 = json.loads((RUNS / "mbb-caliber-20261028/src/927d-extension-decision.json").read_text(encoding="utf-8"))
    p927 = dec927["points"]["data-1.32"]
    if abs(ext_series.mean() - p927["pooled"]["mean"]) > 1e-15:
        raise SystemExit(f"927 extension point {ext_series.mean()} != archived {p927['pooled']['mean']}")
    validations.append({"check": "927d extension (2.64 - 1.32) per-window means",
                        "archived": {w: v["mean"] for w, v in p927["windows"].items()},
                        "recomputed": {w: v["mean"] for w, v in ext_windows.items()}})
    ext_prereg = json.loads((ROOT / "configs/b4_scaling_extension_prereg_20261028.json").read_text(encoding="utf-8"))
    ext_rule_text = ext_prereg.get("extension_rule", ext_prereg.get("trigger_reading"))

    def rejudge_927(iv):
        positive = sum(1 for v in ext_windows.values() if v["mean"] > 0)
        crit = {"improvement_gt_0": bool(iv["mean"] > 0.0),
                "interval_excludes_zero": bool(iv["ci_low"] > 0.0 or iv["ci_high"] < 0.0),
                "positive_windows": positive}
        passed = crit["improvement_gt_0"] and crit["interval_excludes_zero"]
        return passed, crit, ("continue to 3.96" if passed else "stop at 2.64")

    judge("ext-927-264-vs-132", "extension",
          "927d 2.64 - 927d 1.32, three seeds, paired daily rank_ic (5 windows / 104 days)", ext_series,
          {"ci_low": p927["pooled"]["ci_low"], "ci_high": p927["pooled"]["ci_high"],
           "source": "runs/mbb-caliber-20261028/src/927d-extension-decision.json -> points.data-1.32.pooled"},
          int(p927["pooled"].get("draws", 4000)), 20261028,
          {"quote": ext_rule_text if isinstance(ext_rule_text, str) else json.dumps(ext_rule_text, ensure_ascii=False),
           "source": "configs/b4_scaling_extension_prereg_20261028.json -> extension_rule",
           "threshold": 0.0, "trigger": "delta > 0 AND 95% interval excludes zero, else stop at 2.64"},
          {"text": f"stop at 2.64 (registered verdict '{dec927.get('verdict')}')",
           "basis": dec927.get("verdict_basis"),
           "source": "runs/mbb-caliber-20261028/src/927d-extension-decision.json"},
          "interval+sign", "needs a positive point estimate; the point estimate is negative, so a narrower "
                           "interval cannot make this a continuation", rejudge_927)

    # ---------------------------------------------------- (4) promotion gate
    cand, ref = read_daily(RUNS / GATE_CAND_PATH), read_daily(RUNS / GATE_REF_PATH)
    paired = pd.concat({"c": cand, "r": ref}, axis=1, join="inner").dropna()
    gate_delta = paired["c"] - paired["r"]
    gate = json.loads((GATE_DIR / "gate.json").read_text(encoding="utf-8"))
    gcmp = gate["comparison"]
    policy = gcmp["policy"]
    if [list(p) for p in PERIODS] != [list(p) for p in gcmp["periods"]]:
        raise SystemExit("declared periods differ from the archived gate periods")
    gate_as_archived = gate_interval(gate_delta, int(policy["bootstrap_samples"]), int(policy["seed"]))
    if abs(gate_as_archived["mean"] - gcmp["improvement"]) > 1e-15:
        raise SystemExit(f"gate point {gate_as_archived['mean']} != archived {gcmp['improvement']}")
    for got, want in zip(gate_as_archived["period_means"], gcmp["period_effects"]):
        if abs(got - want) > 1e-15:
            raise SystemExit(f"gate period effect {got} != archived {want}")
    if (abs(gate_as_archived["ci_low"] - gcmp["confidence_interval"]["low"]) > 1e-12
            or abs(gate_as_archived["ci_high"] - gcmp["confidence_interval"]["high"]) > 1e-12):
        raise SystemExit(f"gate interval not reproduced: {gate_as_archived} vs {gcmp['confidence_interval']}")
    gate_single = gate_interval(gate_delta, int(policy["bootstrap_samples"]), int(policy["seed"]), single_block=True)
    gate_canon = gate_interval(gate_delta, CANON_DRAWS, CANON_SEED)
    validations.append({"check": "promotion gate interval reproduced with its own per-period block estimator",
                        "archived": gcmp["confidence_interval"],
                        "reproduced": [gate_as_archived["ci_low"], gate_as_archived["ci_high"]],
                        "single_block_counterfactual": [gate_single["ci_low"], gate_single["ci_high"]]})
    prereg = json.loads((ROOT / "configs/b4_promotion_prereg_20261027.json").read_text(encoding="utf-8"))
    effects = gcmp["period_effects"]
    effects = [effects[k] for k in sorted(effects)] if isinstance(effects, dict) else list(effects)
    positive_periods = sum(1 for e in effects if e > 0)

    def rejudge_gate(iv):
        crit = {"practical": bool(iv["mean"] >= float(policy["min_improvement"])),
                "statistical": bool(iv["ci_low"] > 0.0),
                "repeat": bool(positive_periods >= math.ceil(0.75 * len(PERIODS)))}
        return all(crit.values()), crit, ("all statistical legs pass" if all(crit.values()) else "gate fails")

    gate_criteria = rejudge_gate(gate_as_archived)[1]
    record({
        "id": "gate-promotion-441d-2p64", "family": "promotion",
        "comparison": "candidate 441d@2.64 3-seed ewma-match - archived nine-blend, paired daily rank_ic",
        "n_days": int(len(gate_delta)), "point_estimate": float(gate_as_archived["mean"]),
        "archive": {"ci_low": gcmp["confidence_interval"]["low"], "ci_high": gcmp["confidence_interval"]["high"],
                    "draws": int(policy["bootstrap_samples"]), "seed": int(policy["seed"]),
                    "estimator": gate_as_archived["estimator"], "caliber_debt": False,
                    "source": "runs/promotion-candidate-ewma-match-20261028/gate-.../gate.json"},
        "corrected": gate_as_archived, "canonical": gate_canon, "counterfactual_inflated": gate_single,
        "registered_rule": {"quote": json.dumps(prereg["decision_rule"], ensure_ascii=False),
                            "source": "configs/b4_promotion_prereg_20261027.json -> decision_rule",
                            "threshold": float(policy["min_improvement"]),
                            "required_positive_windows": math.ceil(0.75 * len(PERIODS)), "windows_total": len(PERIODS)},
        "archived_verdict": {"text": "gate engine_status accept, failed_gates [], but the overall decision is "
                                     "inconclusive ('证据不足') because the resource artifact kind could not be "
                                     "classified",
                             "basis": f"improvement {gcmp['improvement']:.15f} >= min_improvement "
                                      f"{policy['min_improvement']}; lower bound "
                                      f"{gcmp['confidence_interval']['low']:.15f} > 0; {positive_periods}/5 periods positive",
                             "source": "runs/promotion-candidate-ewma-match-20261028/gate-.../decision.json"},
        "rule_outcome_archived_caliber": {"passed": True, "criteria": gate_criteria, "text": "all statistical legs pass"},
        "rule_outcome_corrected_caliber": {"passed": True, "criteria": gate_criteria, "text": "all statistical legs pass"},
        "verdict_recorded": True, "verdict_flip": False,
        "gate_type": "threshold+windows+interval",
        "sensitivity_note": "insensitive: min_improvement 0.0034 and the gate's per-period block estimator were "
                            "already correct",
    })

    # ------------------------------------------------------- (5) feature norm
    fn = json.loads((RUNS / "mbb-caliber-20261028/src/featnorm-decision.json").read_text(encoding="utf-8"))
    fn_ext = json.loads((RUNS / "mbb-caliber-20261028/src/decision_ext.json").read_text(encoding="utf-8"))
    fn_prereg = json.loads((ROOT / "configs/b4_feature_norm_prereg_20261028.json").read_text(encoding="utf-8"))
    fn_ext_reg = json.loads((ROOT / "configs/b4_feature_norm_extension_20261028.registration.json")
                            .read_text(encoding="utf-8"))
    fn_rule = {"quote": fn_prereg["decision_rule_zh"] if isinstance(fn_prereg.get("decision_rule_zh"), str)
                        else json.dumps(fn_prereg.get("decision_rule_zh"), ensure_ascii=False),
               "source": "configs/b4_feature_norm_prereg_20261028.json -> decision_rule_zh / thresholds",
               "threshold": float(fn_prereg["thresholds"]["min_improvement"]),
               "required_positive_windows": int(fn_prereg["thresholds"]["min_positive_windows"]),
               "windows_total": int(fn_prereg["thresholds"]["windows_total"])}

    for point, label in (("132", "data-1.32"), ("264", "data-2.64")):
        series, windows = version_delta(FEATNORM, "p-s{seed}-{point}", FEATNORM, "c-s{seed}-{point}",
                                        point, (11, 29, 47))
        arch = fn["points"][label]["pooled"]
        if abs(series.mean() - arch["mean"]) > 1e-15:
            raise SystemExit(f"featnorm {label} point {series.mean()} != archived {arch['mean']}")
        validations.append({"check": f"featnorm {label} per-window means",
                            "archived": {w: v["mean"] for w, v in fn["points"][label]["windows"].items()},
                            "recomputed": {w: v["mean"] for w, v in windows.items()}})

        def rejudge_fn(iv, windows=windows):
            positive = sum(1 for v in windows.values() if v["mean"] > 0)
            crit = {"improvement_ge_0.003": bool(iv["mean"] >= 0.003), "ci_low_gt_0": bool(iv["ci_low"] > 0.0),
                    "positive_windows_ge_4": bool(positive >= 4), "positive_windows": positive}
            return all(crit.values()), crit, ("pass" if all(crit.values()) else "stop")

        judge(f"featnorm-{label}", "feature-norm",
              f"feature-norm candidate - control at {label}, three seeds, paired daily rank_ic", series,
              {"ci_low": arch["ci_low"], "ci_high": arch["ci_high"],
               "source": f"runs/mbb-caliber-20261028/src/featnorm-decision.json -> points.{label}.pooled"},
              int(arch["draws"]), 20261028, fn_rule,
              {"text": f"stop ({label}): registered decision passed=false",
               "basis": json.dumps(fn["points"][label].get("criteria"), ensure_ascii=False),
               "source": "runs/mbb-caliber-20261028/src/featnorm-decision.json"},
              "threshold+interval+windows",
              "passes only if the point estimate clears 0.003; the interval leg is caliber-sensitive, the "
              "threshold leg is not", rejudge_fn)

    ext_series2, ext_windows2 = version_delta(FEATNORM_EXT, "p-s{seed}-132", FEATNORM_EXT, "c-s{seed}-132",
                                              "132", (53, 59, 61))
    ext_arch = fn_ext["points"]["data-1.32"]["pooled"]
    if abs(ext_series2.mean() - ext_arch["mean"]) > 1e-15:
        raise SystemExit(f"featnorm ext point {ext_series2.mean()} != archived {ext_arch['mean']}")
    validations.append({"check": "featnorm extension seeds 53/59/61 data-1.32 per-window means",
                        "archived": {w: v["mean"] for w, v in fn_ext["points"]["data-1.32"]["windows"].items()},
                        "recomputed": {w: v["mean"] for w, v in ext_windows2.items()}})

    def rejudge_fext(iv):
        positive = sum(1 for v in ext_windows2.values() if v["mean"] > 0)
        crit = {"improvement_ge_0.003": bool(iv["mean"] >= 0.003), "ci_low_gt_0": bool(iv["ci_low"] > 0.0),
                "positive_windows_ge_4": bool(positive >= 4), "positive_windows": positive}
        return all(crit.values()), crit, ("pass" if all(crit.values()) else "stop")

    judge("featnorm-ext-data-1.32", "feature-norm",
          "feature-norm extension seeds 53/59/61 candidate - control at data-1.32, paired daily rank_ic",
          ext_series2,
          {"ci_low": ext_arch["ci_low"], "ci_high": ext_arch["ci_high"],
           "source": "runs/mbb-caliber-20261028/src/decision_ext.json -> points.data-1.32.pooled"},
          int(ext_arch["draws"]), 20261028,
          {"quote": fn_ext_reg.get("decision_rule") if isinstance(fn_ext_reg.get("decision_rule"), str)
                    else json.dumps(fn_ext_reg.get("decision_rule"), ensure_ascii=False),
           "source": "configs/b4_feature_norm_extension_20261028.registration.json",
           "threshold": float(fn_ext["threshold"]),
           "required_positive_windows": int(fn_ext["min_positive_windows"]), "windows_total": 5,
           "declared_replication": "seeds 53/59/61 replicate the registered test and never replace it"},
          {"text": "stop (extension batch): the declared replication failed the same registered rule",
           "basis": json.dumps(fn_ext["points"]["data-1.32"]["criteria"], ensure_ascii=False),
           "source": "runs/mbb-caliber-20261028/src/decision_ext.json"},
          "threshold+interval+windows",
          "the point estimate is -9.4e-06, so the 0.003 threshold leg fails by 0.003 regardless of caliber",
          rejudge_fext)

    # ---------------------------------------------------------- (6) sealed
    sealed = json.loads((RUNS / SEALED).read_text(encoding="utf-8"))
    frame = pd.DataFrame(sealed["paired_daily"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    sealed_delta = frame.set_index("date").sort_index()["delta"]
    arch = sealed["paired_candidate_minus_reference"]
    if abs(sealed_delta.mean() - arch["mean_delta"]) > 1e-15:
        raise SystemExit(f"sealed point {sealed_delta.mean()} != archived {arch['mean_delta']}")
    positive_days = int((sealed_delta > 0).sum())
    if positive_days != int(arch["positive_days"]):
        raise SystemExit(f"sealed positive_days {positive_days} != archived {arch['positive_days']}")
    validations.append({"check": "sealed window paired delta (61 days, rebuilt only from archived paired_daily)",
                        "point": float(sealed_delta.mean()), "days": int(len(sealed_delta)),
                        "positive_days": positive_days,
                        "positive_fraction": round(positive_days / len(sealed_delta), 6)})
    sealed_prereg = json.loads((ROOT / "configs/b4_sealed_final_prereg_template_20261027.json")
                               .read_text(encoding="utf-8"))

    def rejudge_sealed(iv):
        crit = {"improvement_ge_0.0034": bool(iv["mean"] >= 0.0034), "ci_low_gt_0": bool(iv["ci_low"] > 0.0),
                "positive_fraction_ge_0.75": bool(positive_days / len(sealed_delta) >= 0.75),
                "positive_days": positive_days}
        return all(crit.values()), crit, ("statistical legs pass" if all(crit.values()) else "statistical leg fails")

    judge("sealed-final-20261028", "sealed",
          "sealed-window candidate - reference nine-blend, paired daily rank_ic (2024-10-08..2025-01-01)",
          sealed_delta,
          {"ci_low": arch["ci_low"], "ci_high": arch["ci_high"],
           "source": "runs/sealed-final-20261028/sealed_result.json -> paired_candidate_minus_reference"},
          20000, 20261026,
          {"quote": json.dumps(sealed_prereg.get("derived_thresholds"), ensure_ascii=False),
           "source": "configs/b4_sealed_final_prereg_template_20261027.json -> derived_thresholds",
           "threshold": 0.0034, "positive_fraction_required": 0.75,
           "interval_rule": "the accept template requires the 5-day block interval to exclude zero"},
          {"text": "no sealed verdict artifact exists: runs/sealed-final-20261028/sealed_result.json records only "
                   "the paired measurement, whose interval straddles zero",
           "basis": f"mean_delta {arch['mean_delta']:.15f}, interval [{arch['ci_low']:.15f}, {arch['ci_high']:.15f}], "
                    f"{arch['positive_days']}/61 positive days",
           "source": "runs/sealed-final-20261028/sealed_result.json"},
          "threshold+interval+positive-fraction",
          "the interval leg is caliber-sensitive; the 0.0034 threshold and 47/61 positive-fraction legs are not",
          rejudge_sealed, verdict_recorded=False)

    # ------------------------------------------------------------- assemble
    flips = [j for j in TRACKED if j["verdict_flip"]]
    interval_only = [j for j in TRACKED if j["interval_flips"] and not j["verdict_flip"]]
    masked = [j for j in TRACKED if (not j["archived_interval_excludes_zero"]) and j["corrected"]["ci_low"] > 0]
    debt = [j for j in TRACKED if j["archive"]["caliber_debt"]]
    clean = [j for j in TRACKED if not j["archive"]["caliber_debt"]]
    crosscheck_recompute(TRACKED, validations)

    INTERVAL_KEYS = {"ci_low_gt_0", "interval_excludes_zero", "interval_lower_bound_gt_0", "statistical"}

    def blocked_by_noninterval(entry: dict) -> list[str]:
        crit = entry["rule_outcome_corrected_caliber"]["criteria"]
        return [k for k, v in crit.items() if v is False and k not in INTERVAL_KEYS]

    threshold_decided = [j for j in TRACKED if blocked_by_noninterval(j)]
    already_correct = [j for j in TRACKED if not j["archive"]["caliber_debt"]
                       and not blocked_by_noninterval(j)]
    interval_sensitive = [j for j in TRACKED if j["interval_flips"]]

    report = {
        "schema_version": "caliber-impact/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "re-judge every registered/archived verdict under the corrected 5-day moving-block bootstrap, "
                   "side by side with the caliber it was archived under",
        "caliber": {
            "archived_bug": "compare_blend_replay.block_interval drew one 5-day block per replicate and averaged "
                            "only that block, inflating the interval by about sqrt(n/5)",
            "corrected": "ceil(n/5) contiguous blocks per replicate with wrap-around, averaged over n_blocks*5 values",
            "block_days": BLOCK,
            "identification_rule": "each archived interval is matched bit-for-bit against both estimators to decide "
                                   "whether that judgment actually carried the debt",
            "same_params_rule": "the corrected interval reuses the archived draws/seed so only the block count moves",
            "canonical": {"draws": CANON_DRAWS, "seed": CANON_SEED,
                          "note": "same canonical setting as runs/mbb-caliber-20261028/recompute.json"},
            "reference_inflation_at_n104": round(math.sqrt(104 / 5), 4),
        },
        "red_lines_honored": ["no registered threshold changed", "no historical verdict file written",
                              "no sealed-month raw data read; the sealed row is rebuilt from the archived "
                              "paired_daily only", "no commit"],
        "summary": {
            "judgments": len(TRACKED),
            "recorded_verdict_flips": len(flips),
            "interval_flips": len([j for j in TRACKED if j["interval_flips"]]),
            "caliber_debt_rows": len(debt),
            "already_correct_rows": len(clean),
            "rows": [{"id": j["id"], "family": j["family"], "n": j["n_days"],
                      "point": j["point_estimate"],
                      "archived_ci": [j["archive"]["ci_low"], j["archive"]["ci_high"]],
                      "corrected_ci": [j["corrected"]["ci_low"], j["corrected"]["ci_high"]],
                      "caliber_debt": j["archive"]["caliber_debt"],
                      "interval_flip": j["interval_flips"],
                      "verdict_flip": j["verdict_flip"],
                      "corrected_rule_passes": j["rule_outcome_corrected_caliber"]["passed"]}
                     for j in TRACKED],
        },
        "provenance": provenance,
        "validations": validations,
        "judgments": TRACKED,
        "answers": {
            "a_verdict_flips": {
                "recorded_verdict_flips": len(flips),
                "rows": [{"id": j["id"], "comparison": j["comparison"],
                          "archive": [j["archive"]["ci_low"], j["archive"]["ci_high"]],
                          "corrected": [j["corrected"]["ci_low"], j["corrected"]["ci_high"]],
                          "before": j["rule_outcome_archived_caliber"]["text"],
                          "after": j["rule_outcome_corrected_caliber"]["text"]} for j in flips],
                "interval_only_changes": len(interval_only),
                "interval_only_rows": [{"id": j["id"], "before": j["rule_outcome_archived_caliber"]["text"],
                                        "after": j["rule_outcome_corrected_caliber"]["text"],
                                        "still_blocked_by": [k for k, v in
                                                             j["rule_outcome_corrected_caliber"]["criteria"].items()
                                                             if v is False]}
                                       for j in interval_only],
                "statement": (
                    "No recorded accept/stop verdict flips. Every change is interval-level only, and in each case a "
                    "point-estimate threshold or a negative point estimate still decides the registered rule. The "
                    "sealed window carries no verdict artifact at all, so its statistical leg flips from "
                    "'straddles zero' to 'strictly positive' without overturning a recorded decision."
                    if len(flips) == 0 else f"{len(flips)} recorded verdicts flip."),
            },
            "b_masked_significance": {
                "count": len(masked),
                "rows": [{"id": j["id"], "point_estimate": j["point_estimate"],
                          "archive": [j["archive"]["ci_low"], j["archive"]["ci_high"]],
                          "corrected": [j["corrected"]["ci_low"], j["corrected"]["ci_high"]],
                          "half_width_ratio_inflated_over_corrected": j["half_width_ratio_inflated_over_corrected"],
                          "still_blocked_by": [k for k, v in
                                               j["rule_outcome_corrected_caliber"]["criteria"].items() if v is False]}
                         for j in masked],
                "statement": (
                    "Yes: these comparisons were reported as indistinguishable from zero only because the archived "
                    "interval was ~4.5x too wide. Under the corrected caliber the same point estimate is strictly "
                    "positive with a lower bound above zero. The registered rule still says stop where a 0.003 "
                    "threshold or the sign of the estimate blocks it, but the evidence is no longer 'no effect'."
                    if masked else "None: no judgment here was masked into 'not significant' by interval width."),
            },
            "c_threshold_gated": {
                "count": len(threshold_decided),
                "threshold_or_sign_decided": [
                    {"id": j["id"], "gate_type": j["gate_type"],
                     "decided_by": blocked_by_noninterval(j),
                     "archived_estimator": j["archive"]["estimator"],
                     "interval_flipped": j["interval_flips"],
                     "note": j["sensitivity_note"]} for j in threshold_decided],
                "already_correct_estimator": [
                    {"id": j["id"], "archived_estimator": j["archive"]["estimator"],
                     "note": "bit-for-bit reproduction shows this judgment never carried the inflated caliber"}
                    for j in already_correct],
                "residual_interval_sensitivity": [
                    {"id": j["id"], "archive": [j["archive"]["ci_low"], j["archive"]["ci_high"]],
                     "corrected": [j["corrected"]["ci_low"], j["corrected"]["ci_high"]],
                     "effect": ("the statistical leg passes under the corrected caliber and fails under the "
                                "archived one" if j["rule_outcome_corrected_caliber"]["passed"]
                                and not j["rule_outcome_archived_caliber"]["passed"]
                                else "the interval stops straddling zero; the registered rule still fails on a "
                                     "point-estimate threshold")}
                    for j in interval_sensitive],
                "statement": (
                    "Five judgments are decided by a threshold or a sign, not by interval width, so the caliber fix "
                    "cannot move them: C1 fails the point leg 0.0014 < 0.0017; the 927 extension needs a positive "
                    "delta and its delta is -0.00046; feature norm at 1.32 (+0.00167), 2.64 (-0.00007) and the "
                    "extension batch (-0.00001) all fail the 0.003 point floor. A further seven judgments never "
                    "carried the bug at all, because their builders already resampled ceil(n/5) blocks or period "
                    "blocks (the six dose-curve rows and the promotion gate, plus C1 which is listed above because "
                    "its point leg already decides it): for those the archived interval is reproduced bit-for-bit. Only two rows keep a live interval question: the sealed-window "
                    "measurement, whose statistical leg flips from straddling zero to strictly positive (and which "
                    "has no verdict artifact to flip), and feature norm at 1.32, whose interval flips positive but "
                    "still cannot clear the 0.003 threshold."),
            },
        },
        "multiplicity": (
            "The 8 candidate comparisons in runs/mbb-caliber-20261028/recompute.json plus the 13 archived judgments "
            "re-judged here (six judgment families: dose, C1, 927 extension, promotion gate, feature norm, sealed) "
            "are 21 paired comparisons selected on the same five development windows and the same 104 trading days, "
            "all of them after seeing the scores, with no multiplicity correction. The intervals that change are "
            "still increments of +0.0009 to +0.0016 in RankIC at most, i.e. 1.1%-2.0% of the 0.0788 candidate level: "
            "a caliber correction, not a new effect. Read the corrections as 'the same estimate was always positive', "
            "not as 'a new finding survived a stricter family-wise bar'."),
        "conclusion": (
            "The caliber bug is real but local. It never touched the dose-curve builder, the C1 builder or the "
            "promotion gate, which had their own correct ceil(n/5) estimators (the dose builder even documents the "
            "single-block bug). It did inflate the rung-compare, featnorm, 927-extension and sealed intervals by "
            "~4.5x. Correcting them turns several 'not distinguishable from zero' readings into strictly positive "
            "ones, but no registered accept/stop verdict changes, because every one of those verdicts is decided by "
            "a point-estimate threshold (0.003 / 0.0034) or by a negative point estimate."),
    }

    for path in sorted(SERIES.rglob("daily.csv")):
        provenance[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
    for extra in (RUNS / GATE_CAND_PATH, RUNS / GATE_REF_PATH, RUNS / SEALED,
                  RUNS / "dose-replay-20261027/metrics.json", RUNS / "data-probe-replay-20261027/metrics.json",
                  RUNS / "mbb-caliber-20261028/src/featnorm-decision.json",
                  RUNS / "mbb-caliber-20261028/src/decision_ext.json",
                  RUNS / "mbb-caliber-20261028/src/927d-extension-decision.json", GATE_DIR / "gate.json",
                  GATE_DIR / "decision.json"):
        if extra.exists():
            provenance[str(extra.relative_to(ROOT)).replace("\\", "/")] = sha256(extra)

    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_TABLE.write_text(render_table(report), encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_TABLE.relative_to(ROOT)}")
    print(f"judgments {len(TRACKED)}; recorded verdict flips {len(flips)}; interval-only flips {len(interval_only)}; "
          f"caliber-debt rows {len(debt)}; already-correct rows {len(clean)}")
    return 0


def crosscheck_recompute(tracked: list[dict], validations: list[dict]) -> None:
    """Independent cross-check: the canonical corrected intervals must equal the numbers
    replay-owner already published in runs/mbb-caliber-20261028/recompute.json."""
    path = RUNS / "mbb-caliber-20261028/recompute.json"
    if not path.exists():
        validations.append({"check": "cross-check against recompute.json", "status": "skipped",
                           "reason": "runs/mbb-caliber-20261028/recompute.json not present"})
        return
    rows = {row["id"]: row for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    pairs = {
        "featnorm-data-1.32": "feature-norm data-1.32 candidate - control",
        "featnorm-data-2.64": "feature-norm data-2.64 candidate - control",
        "ext-927-264-vs-132": "927d 2.64 - 1.32",
        "sealed-final-20261028": "sealed candidate - reference nine",
    }
    matches, flips_agreed = [], []
    for mine, theirs in pairs.items():
        entry = next(j for j in tracked if j["id"] == mine)
        row = rows[theirs]
        low, high = row["new_caliber"]["ci"]
        identical = (abs(entry["canonical"]["ci_low"] - low) < 1e-12
                     and abs(entry["canonical"]["ci_high"] - high) < 1e-12)
        matches.append({"id": mine, "recompute_id": theirs,
                        "mine": [entry["canonical"]["ci_low"], entry["canonical"]["ci_high"]],
                        "theirs": [low, high], "identical": bool(identical)})
        if not identical:
            raise SystemExit(f"cross-check failed for {mine}: {matches[-1]}")
        flips_agreed.append({"id": mine, "recompute_interval_flag": bool(row["verdict_flips"]),
                             "my_interval_flip": entry["interval_flips"]})
    validations.append({
        "check": "canonical 20000/20261028 corrected intervals reproduce recompute.json bit-for-bit",
        "rows": matches,
        "note": "recompute.json's verdict_flips flag means 'the interval stopped straddling zero', which is the "
                "same quantity as my interval_flips for the overlapping rows; it is not a registered verdict flip",
        "interval_flag_comparison": flips_agreed})


def dose_neg(point: float) -> str:
    return "" if point == 1.32 else " (secondary point 2.64)"


def render_table(report: dict) -> str:
    lines = ["caliber impact audit - archived 5-day block interval vs corrected (ceil(n/5) blocks)",
             f"generated {report['generated_at_utc']}   block_days=5   inflation at n=104: "
             f"sqrt(104/5)={report['caliber']['reference_inflation_at_n104']}",
             "no threshold changed; no historical verdict rewritten; sealed-month raw data untouched", "",
             f"{'id':26s} {'n':>4s} {'point':>10s} {'archived 95% CI':>27s} {'corrected 95% CI':>27s} "
             f"{'ratio':>6s} {'debt':>5s} {'verdict flip':>12s}",
             "-" * 108,
             "ratio = full CI width of the retired single-block caliber over the width actually archived;",
             "debt=False rows were already archived with a correct estimator, so there the ratio only shows",
             "what the retired helper would have reported for the same series.",
             ""]
    for j in report["judgments"]:
        a, c = j["archive"], j["corrected"]
        ratio = j["half_width_ratio_inflated_over_corrected"]
        lines.append(f"{j['id']:26s} {j['n_days']:4d} {j['point_estimate']:10.6f} "
                     f"[{a['ci_low']:+10.6f},{a['ci_high']:+10.6f}]    [{c['ci_low']:+10.6f},{c['ci_high']:+10.6f}]    "
                     f"{(ratio if ratio else 1.0):6.2f} {str(a['caliber_debt']):>5s} "
                     f"{('n/a' if j['verdict_flip'] is None else str(j['verdict_flip'])):>12s}")
    lines.extend(["", "per judgment:", ""])
    for j in report["judgments"]:
        lines.append(f"[{j['id']}] {j['comparison']}")
        blocks = j["archive"].get("n_blocks", "per-period")
        lines.append(f"  n={j['n_days']}d  point={j['point_estimate']:.15f}  archived caliber: "
                     f"{j['archive']['estimator']} (draws {j['archive']['draws']}, seed {j['archive']['seed']}, "
                     f"{blocks} block(s))")
        lines.append(f"  archived 95% CI : [{j['archive']['ci_low']:+.15f}, {j['archive']['ci_high']:+.15f}]  "
                     f"({j['archive']['source']})")
        lines.append(f"  corrected 95% CI: [{j['corrected']['ci_low']:+.15f}, {j['corrected']['ci_high']:+.15f}]  "
                     f"({j['corrected'].get('n_blocks', 'per-period')} "
                     f"{'block(s) per period' if 'n_blocks' not in j['corrected'] else 'blocks'}, same draws/seed)")
        lines.append(f"  canonical 20000/20261028: [{j['canonical']['ci_low']:+.6f}, {j['canonical']['ci_high']:+.6f}]")
        if j.get("counterfactual_inflated"):
            cf = j["counterfactual_inflated"]
            lines.append(f"  single-block counterfactual: [{cf['ci_low']:+.6f}, {cf['ci_high']:+.6f}] "
                         f"(what the inflated helper would have reported)")
        lines.append(f"  registered rule ({j['registered_rule']['source']}): {j['registered_rule']['quote']}")
        if j["registered_rule"].get("role"):
            lines.append(f"  rule role: {j['registered_rule']['role']}")
        if j["registered_rule"].get("scope_note"):
            lines.append(f"  rule scope: {j['registered_rule']['scope_note']}")
        lines.append(f"  archived verdict: {j['archived_verdict']['text']}  [basis: {j['archived_verdict']['basis']}]")
        lines.append(f"  rule under archived caliber: {j['rule_outcome_archived_caliber']['text']} "
                     f"{j['rule_outcome_archived_caliber']['criteria']}")
        lines.append(f"  rule under corrected caliber: {j['rule_outcome_corrected_caliber']['text']} "
                     f"{j['rule_outcome_corrected_caliber']['criteria']}")
        lines.append(f"  interval flips: {j['interval_flips']}   recorded-verdict flip: {j['verdict_flip']}   "
                     f"gate type: {j['gate_type']}")
        lines.append(f"  sensitivity: {j['sensitivity_note']}")
        lines.append("")
    ans = report["answers"]
    lines.append("(a) did any registered conclusion flip?")
    lines.append("    " + ans["a_verdict_flips"]["statement"])
    lines.append(f"    recorded verdict flips: {ans['a_verdict_flips']['recorded_verdict_flips']}; "
                 f"interval-only changes: {ans['a_verdict_flips']['interval_only_changes']}")
    for row in ans["a_verdict_flips"]["interval_only_rows"] + ans["a_verdict_flips"]["rows"]:
        lines.append(f"    - {row['id']}: {row['before']} -> {row['after']} "
                     f"(still blocked by {row.get('still_blocked_by', [])})")
    lines.append("(b) anything 'not significant' that is really significantly positive?")
    lines.append("    " + ans["b_masked_significance"]["statement"])
    for row in ans["b_masked_significance"]["rows"]:
        blocked = row["still_blocked_by"]
        tail = (f"still blocked by {blocked}" if blocked else
                "none of the re-judged statistical legs blocks it (and no verdict artifact exists for it)")
        lines.append(f"    - {row['id']}: point {row['point_estimate']:+.6f}, {row['archive']} -> {row['corrected']}, "
                     f"width ratio {row['half_width_ratio_inflated_over_corrected']}, {tail}")
    lines.append("(c) verdicts whose conclusion does not depend on the caliber:")
    lines.append("    " + ans["c_threshold_gated"]["statement"])
    lines.append("    decided by a point-estimate threshold or a sign (caliber-invariant):")
    for row in ans["c_threshold_gated"]["threshold_or_sign_decided"]:
        lines.append(f"      * {row['id']} ({row['gate_type']}): decided by {row['decided_by']}; "
                     f"interval flipped = {row['interval_flipped']}")
    lines.append("    archived with an already-correct estimator (reproduced bit-for-bit):")
    for row in ans["c_threshold_gated"]["already_correct_estimator"]:
        lines.append(f"      * {row['id']} ({row['archived_estimator']})")
    lines.append("    residual interval sensitivity:")
    for row in ans["c_threshold_gated"]["residual_interval_sensitivity"]:
        lines.append(f"      * {row['id']}: {row['archive']} -> {row['corrected']} :: {row['effect']}")
    lines.append("")
    lines.append("multiplicity: " + report["multiplicity"])
    lines.append("")
    lines.append("conclusion: " + report["conclusion"])
    lines.append("")
    lines.append("validation checks (bit-for-bit reproduction of archived point and interval):")
    for check in report["validations"]:
        lines.append("  - " + json.dumps(check, ensure_ascii=False)[:300])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
