"""Training-amount budget-point analysis: is the fixed recipe undertrained?

The question this script answers is narrow and single-factor: at the fixed MLPLOB v3
60-day CSZ recipe, does spending more optimizer budget (1.32 -> 2.64 -> 5.28 effective
epochs across one continuous trajectory) improve the competition metrics on the five
already-viewed development months? Only the optimizer budget differs; trunk, data,
labels, loss, sampling, learning-rate rule and evaluation are frozen.

What it does
  * pools each budget point's replay windows into one sequence and re-evaluates it with
    the single eval-v2 evaluator at 0 / 2 / 5 / 10 bp single-side cost, 5bp being the
    locked stress scenario (never averaged, never relaxed);
  * writes gate-compatible merged objects in the shape of the pre-registered baseline
    object: ``metrics.json`` with ``metrics_0bp``/``metrics_5bp`` plus
    ``daily_{0,2,5,10}bp.csv``;
  * reconciles the declared effective-epoch budget against the measured exposure report
    so a configuration cannot claim more passes than the run actually covered;
  * reports every ordered pair with a paired-by-trading-day difference, a 5-day
    consecutive-block interval under the registered policy, and per-period signs.

What it never does
  * train, select or promote a model; every threshold is read from the registered
    pre-registration and is never recomputed, relaxed or replaced here;
  * read the sealed interval ``[2024-10-08, 2025-01-01)`` - any window entering it is
    refused before a metric is computed;
  * write under ``runs/`` - every artifact goes to ``--output``; a dry run stays in
    ``tmp/``;
  * compute a metric anywhere except ``jinyiquant.evaluation.evaluate``.

Usage
    python scripts/training_budget_analysis.py --output tmp/t5-dryrun \
        --point anchor-1.32=<replay dir>[+<replay dir>...] \
        --point double-2.64=<replay dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cost_curve_eval import (  # noqa: E402  (sibling script: one source for windows/rates)
    PRIMARY_RATE,
    RATES,
    SEALED,
    WINDOWS,
    evaluation_config,
    num,
    rel,
)

from jinyiquant.comparison import compare_evaluation_objects  # noqa: E402
from jinyiquant.evaluation import (  # noqa: E402
    EVALUATION_VERSION,
    breakeven_single_side_rate,
    evaluate,
)
from jinyiquant.io import file_sha256, utc_now, write_json  # noqa: E402
from jinyiquant.trajectory import (  # noqa: E402
    effective_epochs_for_steps,
    load_plan,
    reconcile_exposure,
    steps_for_effective_epochs,
)

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "configs/b4_promotion_prereg_20261027.json"
TRAJECTORY_PLAN = ROOT / "configs/training_trajectory_20261027_plan.json"
VOLATILITY = ROOT / "runs/b4-volatility-20261027/20261027T064907Z_a79926d3e455/volatility.json"
WINDOW_SPECS = tuple((name, start, end) for name, start, end, _ in WINDOWS)
SECONDARY = ("ic_mean", "ic_ir", "long_short_sharpe", "long_short_return_mean",
             "min_daily_long_short_return", "turnover_mean")


def _fail(message: str) -> None:
    raise SystemExit(f"refused: {message}")


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Magnitude ratio against the anchor; undefined rather than infinite when the anchor is 0."""
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(numerator / denominator)


def resolve_output(path: Path) -> Path:
    """Every artifact stays outside runs/; a research run archive is never written here."""
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    runs = (ROOT / "runs").resolve()
    if resolved == runs or runs in resolved.parents:
        _fail(f"--output {resolved} is inside runs/; this entry writes analysis artifacts only")
    return resolved


def parse_points(values: list[str]) -> list[dict]:
    """Parse ``label=dir[+dir]`` specs, keeping the declared order (first one is the anchor)."""
    points, seen = [], set()
    for value in values:
        label, _, dirs = value.partition("=")
        label = label.strip()
        if not label or not dirs:
            _fail(f"--point expects label=dir[+dir], got {value!r}")
        if label in seen:
            _fail(f"duplicate budget point label {label!r}")
        seen.add(label)
        directories = [(ROOT / item.strip() if not Path(item.strip()).is_absolute()
                        else Path(item.strip())) for item in dirs.split("+") if item.strip()]
        if not directories:
            _fail(f"budget point {label!r} names no replay directory")
        points.append({"label": label, "directories": directories})
    if len(points) < 1:
        _fail("at least one --point is required")
    return points


def verify_window_definitions(path: Path) -> dict:
    """The five declared windows must match the registered volatility measurement."""
    document = json.loads(path.read_text(encoding="utf-8"))
    declared = {window["name"]: window for window in document.get("windows", [])}
    if set(declared) != {name for name, _, _ in WINDOW_SPECS}:
        _fail(f"{rel(path)} declares windows {sorted(declared)}, expected "
              f"{sorted(name for name, _, _ in WINDOW_SPECS)}")
    targets_sha = {}
    for name, start, end in WINDOW_SPECS:
        window = declared[name]
        if str(window.get("start")) != start or str(window.get("end")) != end:
            _fail(f"window {name}: declared range {window.get('start')}..{window.get('end')} "
                  f"differs from the registered {start}..{end}")
        sha = window.get("targets_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            _fail(f"window {name}: the calibration does not pin targets_sha256")
        targets_sha[name] = sha
    return {"path": rel(path), "sha256": file_sha256(path), "windows": sorted(declared),
            "targets_sha256": targets_sha}


def load_window(name: str, start: str, end: str, directories: list[Path]) -> dict:
    """Read one replay window for one budget point, refusing sealed or out-of-range days."""
    base = None
    for directory in directories:
        candidate = directory / "windows" / name
        if (candidate / "targets.parquet").is_file():
            if base is not None:
                _fail(f"window {name} is provided by more than one replay directory")
            base = candidate
    if base is None:
        _fail(f"window {name} not found in {[rel(d) for d in directories]}")
    targets = pd.read_parquet(base / "targets.parquet")
    keys = targets[["date", "instrument"]].reset_index(drop=True)
    if keys.duplicated().any():
        _fail(f"{name}: duplicate target keys")
    ordered = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    dates = pd.to_datetime(ordered["date"])
    if (dates < pd.Timestamp(start)).any() or (dates >= pd.Timestamp(end)).any():
        _fail(f"{name}: dates outside the declared window {start}..{end}")
    if (dates >= pd.Timestamp(SEALED[0])).any():
        _fail(f"{name}: dates enter the sealed interval; the sealed window must stay unread")
    return {"name": name, "start": start, "end": end, "base": base,
            "keys": ordered[["date", "instrument"]],
            "targets": ordered[["date", "instrument", "target"]],
            "targets_sha256": file_sha256(base / "targets.parquet")}


def read_scores(window: dict, object_id: str) -> np.ndarray:
    label = object_id
    path = window["base"] / object_id / "predictions.parquet"
    if not path.is_file():
        _fail(f"{window['name']}/{label}: predictions.parquet is missing")
    frame = pd.read_parquet(path)
    if set(frame.columns) != {"date", "instrument", "score"}:
        _fail(f"{window['name']}/{label}: unexpected prediction columns {sorted(frame.columns)}")
    merged = frame.merge(window["keys"], on=["date", "instrument"], how="inner",
                         validate="one_to_one").sort_values(["date", "instrument"])
    merged = merged.reset_index(drop=True)
    if len(merged) != len(window["keys"]):
        _fail(f"{window['name']}/{label}: predictions do not cover every target key")
    if not merged[["date", "instrument"]].equals(window["targets"][["date", "instrument"]]):
        _fail(f"{window['name']}/{label}: key order mismatch")
    values = merged["score"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        _fail(f"{window['name']}/{label}: non-finite scores")
    return values


def load_point(point: dict, calibration: dict) -> dict:
    """Pool the five declared windows of one budget point into a single scored sequence."""
    replay = point["object_id"]
    windows = [load_window(name, start, end, point["directories"]) for name, start, end in WINDOW_SPECS]
    for window in windows:
        expected = calibration["targets_sha256"][window["name"]]
        if window["targets_sha256"] != expected:
            _fail(f"{window['name']}: targets_sha256 {window['targets_sha256'][:16]} differs from the "
                  f"registered calibration {expected[:16]}")
    keys = pd.concat([w["keys"] for w in windows], ignore_index=True)
    targets = pd.concat([w["targets"] for w in windows], ignore_index=True)
    keys = keys.sort_values(["date", "instrument"]).reset_index(drop=True)
    targets = targets.sort_values(["date", "instrument"]).reset_index(drop=True)
    if not keys.equals(targets[["date", "instrument"]]) or keys.duplicated().any():
        _fail(f"{replay}: pooled target keys are not unique/consistent")
    pieces = [read_scores(window, replay) for window in windows]
    offsets = np.cumsum([0] + [len(window["keys"]) for window in windows])
    positions = pd.concat([window["keys"].assign(_position=np.arange(len(window["keys"])) + offset)
                           for window, offset in zip(windows, offsets)], ignore_index=True)
    positions = positions.sort_values(["date", "instrument"]).reset_index(drop=True)
    scores = np.concatenate(pieces)[positions["_position"].to_numpy(dtype=int)]
    predictions = keys.assign(score=scores)[["date", "instrument", "score"]]
    return {"label": point["label"], "object_id": replay,
            "windows": [w["name"] for w in windows], "keys": keys,
            "targets": targets, "predictions": predictions, "scores": scores,
            "window_predictions_sha256": {w["name"]: file_sha256(w["base"] / replay / "predictions.parquet")
                                         for w in windows},
            "targets_sha256": {w["name"]: w["targets_sha256"] for w in windows}}


def evaluate_point(point: dict) -> dict:
    """One budget point evaluated at every predeclared single-side rate."""
    results = {}
    for label, rate in RATES:
        metrics, daily, _ = evaluate(point["predictions"], point["targets"],
                                     evaluation_config(rate), return_sections=True,
                                     undefined_constant_portfolios=True)
        if metrics.get("evaluation_version") != EVALUATION_VERSION:
            _fail(f"{point['label']} {label}: evaluation is not {EVALUATION_VERSION}")
        results[label] = {"single_side_rate": rate, "metrics": metrics, "daily": daily}
    return results


def economics(block: dict) -> dict:
    """Gross / cost / net in basis points plus the local turnover proxy, never a score."""
    daily = block["daily"]
    gross = float(daily["long_short_return"].mean())
    cost = float(daily["cost"].mean())
    net = float(daily["net_return"].mean())
    change = float(daily["total_abs_weight_change"].mean())
    return {
        "single_side_rate_bp": block["single_side_rate"] * 1e4,
        "gross_return_bp": gross * 1e4,
        "cost_bp": cost * 1e4,
        "net_return_bp": net * 1e4,
        "turnover_mean_local": float(daily["turnover"].mean()),
        "total_abs_weight_change_mean": change,
        "breakeven_single_side_rate_bp": (None if not np.isfinite(change) or change <= 0
                                          else float(breakeven_single_side_rate(
                                              daily["long_short_return"].to_numpy(dtype=float),
                                              daily["total_abs_weight_change"].to_numpy(dtype=float))) * 1e4),
        "share_days_net_positive": float((daily["net_return"] > 0).mean()),
        "official_turnover_item": {
            "official_value": None,
            "why": "the official turnover term is a cross-submission rank percentile; it cannot be "
                   "computed locally and is never substituted by an in-arm percentile",
            "rate_invariant": True,
        },
    }


def write_object(output: Path, point: dict, evaluated: dict) -> dict:
    """Build one gate-compatible merged object document (metrics_0bp/5bp + daily_{rate}bp.csv)."""
    directory = output / "objects" / point["label"]
    directory.mkdir(parents=True, exist_ok=True)
    daily_hashes, scenarios = {}, {}
    for label, block in evaluated.items():
        daily = block["daily"]
        path = directory / f"daily_{label}.csv"
        daily.to_csv(path, index=False)
        daily_hashes[f"{label}_sha256"] = file_sha256(path)
        scenarios[label] = economics(block)
    metrics_document = {
        "schema_version": 1,
        "object_id": point["label"],
        "evaluation_version": EVALUATION_VERSION,
        "windows": point["windows"],
        "caliber": "the five declared research months pooled into one scored sequence and "
                   "evaluated once per cost scenario, matching the pre-registered baseline object",
        "metrics_0bp": evaluated["0bp"]["metrics"],
        "metrics_5bp": evaluated["5bp"]["metrics"],
        "cost_scenarios": scenarios,
        "daily": daily_hashes,
        "sources": {"window_predictions_sha256": point["window_predictions_sha256"],
                    "targets_sha256": point["targets_sha256"]},
    }
    return metrics_document


def verify_gate_format(directory: Path, label: str) -> dict:
    """Prove the written object is loadable by the pre-registered gate, unchanged.

    The gate re-reads an object archive and re-derives every secondary metric from the
    daily artifact, so a successful load plus recompute is a format guarantee, not a
    verdict. The verdict belongs to the promotion gate run and to independent review.
    """
    from promotion_gate import load_object, recompute_secondary  # noqa: PLC0415

    try:
        loaded = load_object(directory, "candidate")
        values = recompute_secondary(loaded["metrics_0bp"], loaded["daily_0bp"], label)
        recompute_secondary(loaded["metrics_5bp"], loaded["daily_5bp"], label)
    except Exception as exc:  # the gate refuses by design: report, do not swallow
        return {"loadable": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"loadable": True, "source_kind": loaded["source_kind"],
            "days_0bp": int(len(loaded["daily_0bp"])),
            "days_5bp": int(len(loaded["daily_5bp"])),
            "primary_metric_0bp": num(values.get("rank_ic_mean")),
            "secondary_recomputed": True}


def compare_pair(baseline: dict, candidate: dict, policy: dict, periods: list) -> dict:
    """Pairwise readings under the registered object policy; no verdict is drawn here."""
    comparison, decision = compare_evaluation_objects(
        {"object_id": baseline["label"], "metrics": baseline["metrics_0bp"],
         "daily": baseline["daily_0bp"]},
        {"object_id": candidate["label"], "metrics": candidate["metrics_0bp"],
         "daily": candidate["daily_0bp"]},
        policy, periods=periods)
    interval = comparison.get("confidence_interval") or {}
    return {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "paired_improvement": comparison.get("improvement"),
        "block_interval": {"low": interval.get("low"), "high": interval.get("high"),
                           "block_days": policy["block_days"],
                           "bootstrap_samples": policy["bootstrap_samples"],
                           "excludes_zero": bool(interval) and float(interval.get("low", 0)) > 0},
        "paired_day_count": int(comparison.get("paired_day_count") or 0),
        "period_effects": comparison.get("period_effects"),
        "engine_diagnostics": {"failed_gates": decision.get("failed_gates"),
                               "missing_evidence": decision.get("missing_evidence")},
        "secondary": comparison.get("secondary"),
    }


def exposure_section(points: list[dict], plan: dict, overrides: dict) -> dict:
    """Declared budget versus measured exposure; missing evidence is reported, never assumed."""
    atol = float(plan["exposure"]["effective_epochs_atol"])
    minimum_coverage = float(plan["exposure"]["min_coverage"])
    declared = {row["name"]: float(row["effective_epochs"]) for row in plan["budget_points"]}
    segments = {}
    for point in points:
        label = point["label"]
        path = overrides.get(label)
        if path is None:
            for directory in point["directories"]:
                candidate = directory / "exposure.json"
                if candidate.is_file():
                    path = candidate
                    break
        if path is None or not Path(path).is_file():
            segments[label] = {"status": "missing",
                               "why": "no exposure.json was found for this budget point; the declared "
                                      "budget cannot be reconciled with what the trajectory covered"}
            continue
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        missing_fields = [field for field in ("usable_samples", "batch_size", "steps",
                                              "effective_epochs", "coverage", "samples_seen",
                                              "expected_samples_seen")
                          if field not in report]
        if missing_fields:
            segments[label] = {"status": "missing",
                               "source": rel(path),
                               "why": f"the exposure report lacks {missing_fields}; the declared "
                                      f"budget cannot be reconciled with what the run covered"}
            continue
        epochs = declared.get(label)
        if epochs is None:
            segments[label] = {"status": "missing",
                               "why": f"{label} is not a pre-registered budget point name; "
                                      f"declared names are {sorted(declared)}"}
            continue
        expected_steps = steps_for_effective_epochs(epochs, int(report["usable_samples"]),
                                                    int(report["batch_size"]))
        verdict = reconcile_exposure(report, {
            "steps": expected_steps, "effective_epochs": epochs,
            "effective_epochs_atol": atol, "min_coverage": minimum_coverage})
        segments[label] = {
            "status": "passed" if verdict["passed"] else "failed",
            "source": rel(path),
            "declared_effective_epochs": epochs,
            "declared_steps_recomputed": expected_steps,
            "realised_steps": report["steps"],
            "realised_effective_epochs_by_steps": effective_epochs_for_steps(
                int(report["steps"]), int(report["usable_samples"]), int(report["batch_size"])),
            "measured_effective_epochs": report["effective_epochs"],
            "samples_seen": report["samples_seen"],
            "expected_samples_seen": report["expected_samples_seen"],
            "coverage": report["coverage"],
            "reconcile": verdict,
        }
    return {"tolerance": {"effective_epochs_atol": atol, "min_coverage": minimum_coverage},
            "segments": segments}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point", action="append", required=True,
                        help="label=replay_dir[+replay_dir]; the first point is the anchor")
    parser.add_argument("--prereg", type=Path, default=PREREG)
    parser.add_argument("--plan", type=Path, default=TRAJECTORY_PLAN)
    parser.add_argument("--windows-json", type=Path, default=VOLATILITY)
    parser.add_argument("--object", action="append", default=[],
                        help="label=object_id inside each window; defaults to the label")
    parser.add_argument("--exposure", action="append", default=[],
                        help="label=exposure.json override")
    parser.add_argument("--note", default="", help="free-form provenance note for this execution")
    args = parser.parse_args(argv)

    output = resolve_output(args.output)
    points = parse_points(args.point)
    object_ids = {item.split("=", 1)[0]: item.split("=", 1)[1]
                  for item in args.object if "=" in item}
    for point in points:
        point["object_id"] = object_ids.get(point["label"], point["label"])
    plan = load_plan(args.plan)
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))

    output.mkdir(parents=True, exist_ok=True)
    calibration = verify_window_definitions(args.windows_json)

    plan_document = {
        "schema_version": 1,
        "created_at": utc_now(),
        "entry": "scripts/training_budget_analysis.py",
        "note": args.note,
        "question": "does spending more optimizer budget on the frozen recipe improve the "
                    "competition metrics on the five declared research months?",
        "single_factor": "optimizer budget only (declared effective epochs per budget point)",
        "budget_points": [{"label": point["label"], "object_id": point["object_id"],
                           "replays": [rel(d) for d in point["directories"]]} for point in points],
        "declared_effective_epochs": {row["name"]: float(row["effective_epochs"])
                                      for row in plan["budget_points"]},
        "cost_scenarios": {label: rate for label, rate in RATES},
        "primary_stress_scenario": PRIMARY_RATE,
        "prereg": {"path": rel(args.prereg), "sha256": file_sha256(args.prereg),
                   "registration_id": prereg.get("registration_id")},
        "windows": calibration,
        "sealed_interval": list(SEALED),
        "sealed_read": False,
        "thresholds_modified": False,
        "writes_under_runs": False,
    }
    write_json(output / "plan.json", plan_document)

    declared_names = {row["name"] for row in plan["budget_points"]}
    loaded, evaluated, objects, format_checks = [], {}, {}, {}
    for point in points:
        loaded_point = load_point(point, calibration)
        results = evaluate_point(loaded_point)
        loaded.append(loaded_point)
        evaluated[point["label"]] = results
        document = write_object(output, loaded_point, results)
        # A replay that is not a pre-registered budget point can only exercise the code path.
        document["role"] = "budget_point" if point["label"] in declared_names else "code_path_control"
        write_json(output / "objects" / point["label"] / "metrics.json", document)
        objects[point["label"]] = document
        format_checks[point["label"]] = verify_gate_format(output / "objects" / point["label"],
                                                           point["label"])

    daily_frames = {label: {scenario: block["daily"]
                            for scenario, block in scenarios.items()}
                    for label, scenarios in evaluated.items()}
    policy = dict(prereg["object_policy_template"])
    policy["change"] = {"factor": "model.max_steps",
                        "baseline": float(plan["budget_points"][0]["effective_epochs"]),
                        "candidate": float(plan["budget_points"][-1]["effective_epochs"]),
                        "hypothesis": "more optimizer budget on the identical recipe improves the "
                                      "competition metrics on the declared research months"}
    periods = [(start, end) for start, end in prereg["periods"]["declared"]]
    names = [point["label"] for point in points]
    pairs = []
    for index in range(len(names) - 1):
        for later in range(index + 1, len(names)):
            baseline, candidate = names[index], names[later]
            reading = compare_pair(
                {"label": baseline, "metrics_0bp": evaluated[baseline]["0bp"]["metrics"],
                 "daily_0bp": daily_frames[baseline]["0bp"]},
                {"label": candidate, "metrics_0bp": evaluated[candidate]["0bp"]["metrics"],
                 "daily_0bp": daily_frames[candidate]["0bp"]},
                policy, periods)
            reading["consecutive"] = later == index + 1
            reading["reading_semantics"] = (
                "improvement and the block interval are paired-day statistics; the min_improvement "
                "line they are compared with is the promotion gate's, so for a budget-point pair it "
                "measures whether more optimizer budget moves the primary metric further than "
                "measured noise, and for a code_path_control it must stay inside that noise scale. "
                "Neither is a promotion verdict.")
            pairs.append(reading)

    exposures = exposure_section(points, plan, {item.split("=", 1)[0]: item.split("=", 1)[1]
                                                for item in args.exposure if "=" in item})

    summary = {
        label: {
            "role": "budget_point" if label in declared_names else "code_path_control",
            "effective_epochs_declared": exposure.get("declared_effective_epochs"),
            "rank_ic_mean_0bp": num(evaluated[label]["0bp"]["metrics"].get("rank_ic_mean")),
            "scenarios": {scenario: {key: num(value) for key, value in block.items()
                                     if isinstance(value, (int, float)) or value is None}
                          for scenario, block in objects[label]["cost_scenarios"].items()},
            "gross_retention_vs_anchor": _ratio(
                objects[label]["cost_scenarios"]["0bp"]["gross_return_bp"],
                objects[names[0]]["cost_scenarios"]["0bp"]["gross_return_bp"]),
            "turnover_ratio_vs_anchor": _ratio(
                objects[label]["cost_scenarios"]["0bp"]["turnover_mean_local"],
                objects[names[0]]["cost_scenarios"]["0bp"]["turnover_mean_local"]),
        } for label, exposure in ((n, exposures["segments"].get(n, {})) for n in names)
    }

    segments = {
        "paired_comparison": {
            "status": "verified" if pairs and all(p["paired_day_count"] for p in pairs) else "missing",
            "what": "paired-by-trading-day difference, 5-day consecutive-block interval inside each "
                    "declared period, and per-period signs under the registered object policy",
            "evidence": {"pairs": len(pairs),
                         "paired_days": [p["paired_day_count"] for p in pairs],
                         "periods": len(prereg["periods"]["names"])},
        },
        "merged_object": {
            "status": "verified" if all(check["loadable"] for check in format_checks.values())
                      else "failed",
            "what": "gate-compatible merged object: metrics_0bp/metrics_5bp plus daily_{0,2,5,10}bp.csv "
                    "over the five declared months",
            "evidence": {label: {"loadable": check["loadable"], "days": check.get("days_0bp")}
                         for label, check in format_checks.items()},
        },
        "economics": {
            "status": "verified" if all(len(objects[label]["cost_scenarios"]) == len(RATES)
                                        for label in names) else "failed",
            "what": "gross/cost/net in basis points at 0/2/5/10bp plus the local turnover proxy; "
                    "5bp is the locked stress scenario and the official turnover term is not faked",
            "evidence": {"scenarios_per_object": {label: sorted(objects[label]["cost_scenarios"])
                                                  for label in names}},
        },
        "exposure_reconciliation": {
            "status": ("verified" if all(segment["status"] != "missing"
                                         for segment in exposures["segments"].values()) else "missing"),
            "what": "declared effective-epoch budget recomputed into steps against the measured pool, "
                    "then reconciled with the trajectory exposure report",
            "evidence": {label: segment["status"]
                         for label, segment in exposures["segments"].items()},
        },
    }
    analysis = {
        "schema_version": 1,
        "created_at": utc_now(),
        "entry": "scripts/training_budget_analysis.py",
        "note": args.note,
        "evaluation_version": EVALUATION_VERSION,
        "question": plan_document["question"],
        "single_factor": plan_document["single_factor"],
        "coverage": {"days": int(pd.to_datetime(loaded[0]["keys"]["date"]).dt.normalize().nunique()),
                     "periods": prereg["periods"]["names"],
                     "windows": calibration["windows"],
                     "sealed_intersection": 0},
        "objects": objects,
        "gate_format": format_checks,
        "segments": segments,
        "summary": summary,
        "pairs": pairs,
        "exposure": exposures,
        "limits": [
            "the five declared months are already-viewed development data; they measure a budget "
            "effect and cannot confirm a candidate",
            "the local turnover proxy U/2 is direction evidence only; the official turnover term is a "
            "cross-submission rank percentile and is not reproduced here",
            "5bp is a locked local stress scenario; the published score contains no cost term",
            "per-pair readings are evidence, not verdicts: the verdict belongs to the promotion gate "
            "and to the independent verification task",
        ],
    }
    write_json(output / "analysis.json", analysis)

    checked = [name for name, segment in exposures["segments"].items() if segment["status"] != "missing"]
    missing = [name for name, segment in exposures["segments"].items() if segment["status"] == "missing"]
    print(f"points={names} days={analysis['coverage']['days']} periods={len(prereg['periods']['names'])} "
          f"pairs={len(pairs)} objects={rel(output / 'objects')}")
    for pair in pairs:
        interval = pair["block_interval"]
        print(f"  {pair['baseline']} -> {pair['candidate']}: improvement="
              f"{num(pair['paired_improvement'])} interval=[{num(interval['low'])}, "
              f"{num(interval['high'])}]")
    for label in names:
        scenarios = objects[label]["cost_scenarios"]
        print(f"  {label}: rank_ic_mean_0bp={num(evaluated[label]['0bp']['metrics'].get('rank_ic_mean'))} "
              f"gross={scenarios['0bp']['gross_return_bp']:.3f}bp "
              f"net5={scenarios['5bp']['net_return_bp']:.3f}bp "
              f"turnover={scenarios['0bp']['turnover_mean_local']:.4f}")
    print("  segments: " + ", ".join(f"{name}={block['status']}"
                                     for name, block in segments.items()))
    print("  gate_format: " + ", ".join(
        f"{label}={'loadable' if format_checks[label]['loadable'] else 'NOT loadable'}"
        f"({format_checks[label].get('days_0bp', '?')}d)" for label in names))
    print(f"  exposure: reconciled={checked or 'none'} missing={missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
