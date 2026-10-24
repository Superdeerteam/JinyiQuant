"""Final G5 consolidation for the V2 label-contrast round (single seed).

The seed-29 arm was *not* triggered: the pre-registered gate in
``configs/label_contrast_20261024_plan.json`` requires a positive pooled RankIC
difference against V2-CSZ, and both candidates are negative in every development
month.  This revision therefore only consolidates the existing seed-11 evidence:

* three versions x one seed x four windows x two scoring variants (raw official
  labels and style-residualized) in one complete table;
* the eight-timestamp view pooled over the three development months;
* training losses by declared unit (never compared across label transforms);
* the pre-registered verdict for both candidates and the H recommendation.

The earlier revision-1 blocks of ``modified_tlob/label_contrast_20261024.json``
are conserved unchanged at the value level; the update is append-only and the
script refuses to write if any pre-existing top-level block changes.  Nothing is
trained or re-scored here and no markdown report is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.io import digest, file_sha256, read_json, utc_now, write_json  # noqa: E402

EXPERIMENT = "label-contrast-20261024"
DEV_MONTHS = ("nov2023", "jan2024", "mar2024")
WINDOWS = ("july2023", "nov2023", "jan2024", "mar2024")
VERSIONS = ("v2_raw", "v2_center", "v2_60d_csz")
BASELINE = "v2_60d_csz"
CANDIDATES = ("v2_raw", "v2_center")
VARIANTS = ("raw_full", "processed")
TABLE_FIELDS = ("rank_ic_mean", "rank_ic_ir", "ic_mean", "ic_ir", "long_short_return_mean",
                "long_short_sharpe", "turnover_mean", "min_daily_long_short_return")
REPLAY_FIELDS = tuple(field for field in TABLE_FIELDS if field != "ic_ir")


def resolve(path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def json_float(value):
    return None if value is None else float(value)


def committed_sha256(path: Path) -> str | None:
    """Sha256 of the committed blob (LF-normalized), or None when git is unavailable."""
    try:
        relative = path.resolve().relative_to(ROOT)
        payload = subprocess.run(["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        return hashlib.sha256(payload).hexdigest()
    except Exception:  # noqa: BLE001 - provenance is optional, never block the record
        return None


def load_archive(path: Path) -> dict:
    archive = read_json(path)
    required = {"schema_version", "experiment_id", "preregistration", "execution", "versions",
                "windows", "results", "development_months", "eight_time_points",
                "style_residualization", "training_losses", "paired", "decision"}
    missing = required - set(archive)
    if missing:
        raise ValueError(f"archive is missing {sorted(missing)}")
    if archive["experiment_id"] != EXPERIMENT:
        raise ValueError("archive experiment_id differs from this entry point")
    if "final_consolidation" in archive:
        raise ValueError("archive already carries the G5 revision; restore revision 1 from git "
                         "before rebuilding so the provenance chain stays single-valued")
    return archive


def verify_preregistration(archive: dict) -> dict:
    plan_path = resolve(archive["preregistration"]["path"])
    recorded = archive["preregistration"]["sha256"]
    if file_sha256(plan_path) != recorded:
        raise ValueError("pre-registered plan file changed after the archive was written")
    plan = read_json(plan_path)
    if plan["experiment_id"] != EXPERIMENT:
        raise ValueError("plan experiment_id differs from the archive")
    return {"path": str(plan_path.relative_to(ROOT)), "sha256": recorded,
            "decision_framework": plan["decision_framework"],
            "statistics": plan["statistics"]}


def verify_replay(archive: dict) -> dict:
    run_dir = resolve(archive["execution"]["replay_run_dir"])
    run_id = archive["execution"]["replay_run_id"]
    manifest = read_json(run_dir / "manifest.json")
    if manifest["run_id"] != run_id or manifest.get("status") != "completed":
        raise ValueError("replay run identity or status changed")
    summary = read_json(run_dir / "summary_metrics.json")
    recorded = {(row["window"], row["version"]): row for row in summary["responses"]}
    archived = {(row["window"], row["version"]): row for row in archive["results"]
                if row["variant"] == "raw_full"}
    if set(recorded) != set(archived):
        raise ValueError("replay window/version grid differs from the archive")
    worst = 0.0
    for key, row in archived.items():
        for field in REPLAY_FIELDS:
            left, right = row[field], recorded[key][field]
            if (left is None) != (right is None):
                raise ValueError(f"{key}: undefined metric mismatch for {field}")
            if left is not None:
                worst = max(worst, abs(left - right))
    if worst > 1e-12:
        raise ValueError(f"archive raw metrics differ from the replay run ({worst})")
    return {"run_id": run_id, "run_dir": str(run_dir.relative_to(ROOT)),
            "wall_seconds": manifest.get("resources", {}).get("wall_seconds"),
            "max_metric_abs_diff": float(worst), "windows": sorted({key[0] for key in archived})}


def verify_weights(archive: dict) -> dict:
    checked = {}
    for name, entry in archive["execution"]["weights"].items():
        path = resolve(entry["path"])
        digest_value = file_sha256(path)
        if digest_value != entry["sha256"] or digest_value != entry["expected_sha256"]:
            raise ValueError(f"{name}: frozen weight checksum changed")
        checked[name] = {"sha256": digest_value, "role": entry["role"]}
    return checked


def verify_seed_coverage(archive: dict) -> dict:
    """Only seed 11 exists; assert no seed-29 artifact was produced anywhere."""
    decision = archive["decision"]
    seed29 = decision["seed29_recommendation"]
    if seed29["run_seed29_for"]:
        raise ValueError("seed-29 trigger is set but this round has no seed-29 artifacts")
    label_contrast_roots = [ROOT / "runs" / "label-contrast-20261024",
                           ROOT / "runs" / "label-contrast-eval-20261024"]
    stray = sorted(str(path.relative_to(ROOT)) for root in label_contrast_roots
                   for path in root.rglob("*seed29*"))
    stray += sorted(str(path.relative_to(ROOT)) for path in (ROOT / "configs").glob("*label_contrast*seed29*"))
    if stray:
        raise ValueError(f"unexpected label-contrast seed-29 artifacts: {stray[:5]}")
    training = archive["training_losses"]["per_arm"]
    seeds = {}
    for name, path in (("v2_raw", ROOT / "runs" / "label-contrast-20261024" / "raw" / "config.json"),
                       ("v2_center", ROOT / "runs" / "label-contrast-20261024" / "center" / "config.json"),
                       (BASELINE, ROOT / "runs" / "model-versions" / "v2-cloud-20261023" / "60d" / "config.json")):
        if not path.is_file():
            raise ValueError(f"missing run config {path}")
        seeds[name] = int(read_json(path)["seed"])
    if set(seeds.values()) != {11}:
        raise ValueError(f"seed coverage changed: {seeds}")
    return {"seeds": {name: [seeds[name]] for name in VERSIONS},
            "seed_count": 1, "seed29_trained": False,
            "training_losses": {name: {"loss": training[name]["loss"],
                                       "loss_units": training[name]["loss_units"]}
                                for name in training}}


def complete_table(archive: dict) -> list[dict]:
    rows = []
    for window in WINDOWS:
        for version in VERSIONS:
            for variant in VARIANTS:
                matches = [row for row in archive["results"]
                           if row["window"] == window and row["version"] == version
                           and row["variant"] == variant]
                if len(matches) != 1:
                    raise ValueError(f"{window}/{version}/{variant}: expected one result row")
                row = matches[0]
                rows.append({"window": window, "version": version,
                             "seed": 11, "variant": variant,
                             "rank_ic_mean": json_float(row["rank_ic_mean"]),
                             "rank_ic_ir": json_float(row["rank_ic_ir"]),
                             "ic_mean": json_float(row["ic_mean"]),
                             "ic_ir": json_float(row["ic_ir"]),
                             "long_short_return_mean": json_float(row["long_short_return_mean"]),
                             "long_short_sharpe": json_float(row["long_short_sharpe"]),
                             "turnover_mean": json_float(row["turnover_mean"]),
                             "min_daily_long_short_return": json_float(
                                 row["min_daily_long_short_return"]),
                             "worst_day": row["worst_day"],
                             "day_count": int(row["day_count"]),
                             "cross_section_count": int(row["cross_section_count"]),
                             "sample_count": int(row["sample_count"])})
    return rows


def development_month_table(archive: dict) -> list[dict]:
    rows = []
    for version in VERSIONS:
        for variant in VARIANTS:
            key = f"{version}/{variant}"
            entry = archive["development_months"]["pooled_metrics"][key]
            rows.append({"version": version, "seed": 11, "variant": variant,
                         "months": list(DEV_MONTHS),
                         "rank_ic_mean": json_float(entry["rank_ic_mean"]),
                         "rank_ic_ir": json_float(entry["rank_ic_ir"]),
                         "ic_mean": json_float(entry["ic_mean"]),
                         "ic_ir": json_float(entry["ic_ir"]),
                         "long_short_return_mean": json_float(entry["long_short_return_mean"]),
                         "long_short_sharpe": json_float(entry["long_short_sharpe"]),
                         "turnover_mean": json_float(entry["turnover_mean"]),
                         "min_daily_long_short_return": json_float(
                             entry["min_daily_long_short_return"]),
                         "worst_day": entry["worst_day"], "day_count": int(entry["day_count"])})
    return rows


def style_before_after(archive: dict) -> list[dict]:
    rows = []
    for window in WINDOWS:
        for version in VERSIONS:
            raw = next(row for row in archive["results"]
                       if row["window"] == window and row["version"] == version
                       and row["variant"] == "raw_full")
            processed = next(row for row in archive["results"]
                             if row["window"] == window and row["version"] == version
                             and row["variant"] == "processed")
            diagnostics = archive["style_residualization"][f"{window}/{version}"]
            retained = (processed["rank_ic_mean"] / raw["rank_ic_mean"]
                        if raw["rank_ic_mean"] else None)
            rows.append({"window": window, "version": version, "seed": 11,
                         "raw_rank_ic_mean": json_float(raw["rank_ic_mean"]),
                         "processed_rank_ic_mean": json_float(processed["rank_ic_mean"]),
                         "rank_ic_retained": json_float(retained),
                         "raw_long_short_return_mean": json_float(raw["long_short_return_mean"]),
                         "processed_long_short_return_mean": json_float(
                             processed["long_short_return_mean"]),
                         "raw_turnover_mean": json_float(raw["turnover_mean"]),
                         "processed_turnover_mean": json_float(processed["turnover_mean"]),
                         "mean_style_explained": diagnostics["mean_style_explained"],
                         "clipped_fraction": diagnostics["clipped_fraction"],
                         "exposure_dropped_fraction": diagnostics["exposure_dropped_fraction"]})
    return rows


def eight_time_point_summary(archive: dict) -> list[dict]:
    """Mean of the three development-month clock means per version and variant."""
    rows = []
    for version in VERSIONS:
        for variant in VARIANTS:
            frames = []
            for entry in archive["eight_time_points"]:
                if entry["version"] == version and entry["variant"] == variant \
                        and entry["window"] in DEV_MONTHS:
                    frames.append(pd.DataFrame(entry["rows"]))
            if len(frames) != len(DEV_MONTHS):
                raise ValueError(f"{version}/{variant}: missing development-month clock rows")
            combined = pd.concat(frames, ignore_index=True).groupby("time", sort=True)[
                ["rank_ic_mean", "long_short_return_mean", "turnover_mean"]].mean()
            for clock, row in combined.iterrows():
                rows.append({"version": version, "variant": variant, "seed": 11,
                             "time": clock,
                             "rank_ic_mean": json_float(row["rank_ic_mean"]),
                             "long_short_return_mean": json_float(row["long_short_return_mean"]),
                             "turnover_mean": json_float(row["turnover_mean"])})
    return rows


def paired_summary(archive: dict) -> dict:
    result = {}
    for candidate in CANDIDATES:
        key = f"{candidate}-minus-{BASELINE}"
        result[candidate] = {}
        for variant in ("raw_full", "processed"):
            entry = next(row for row in archive["paired"]
                         if row["pair"] == key and row["variant"] == variant)
            block5 = entry["blocks"]["block_5"]
            block2 = entry["blocks"]["block_2"]
            result[candidate][variant] = {
                "pooled_mean_delta": entry["pooled"],
                "block5": {metric: block5["all"][metric] for metric in
                           ("rank_ic", "long_short_return", "turnover")},
                "block2": {metric: block2["all"][metric] for metric in
                           ("rank_ic", "long_short_return", "turnover")},
                "per_month_rank_ic_delta": {month: entry["per_month_mean_delta"][month]["rank_ic"]
                                            for month in DEV_MONTHS},
                "day_counts": entry["day_counts"]}
    return result


def evidence_strength(archive: dict, paired: dict) -> dict:
    variants = {}
    for candidate in CANDIDATES:
        rank_ic = paired[candidate]["raw_full"]["block5"]["rank_ic"]
        variants[candidate] = {
            "pooled_rank_ic_delta": rank_ic["mean"],
            "block5_ci": [rank_ic["ci_low"], rank_ic["ci_high"]],
            "block5_half_width": (rank_ic["ci_high"] - rank_ic["ci_low"]) / 2.0,
            "months_negative": sum(1 for value in
                                   paired[candidate]["raw_full"]["per_month_rank_ic_delta"].values()
                                   if value < 0),
            "months_total": len(DEV_MONTHS),
            "positive_day_fraction": (
                paired[candidate]["raw_full"]["day_counts"]["rank_ic"]["positive_days"]
                / sum(paired[candidate]["raw_full"]["day_counts"]["rank_ic"][key]
                      for key in ("positive_days", "negative_days", "zero_days"))),
            "style_view_same_sign": bool(
                (paired[candidate]["processed"]["block5"]["rank_ic"]["mean"] < 0)
                == (rank_ic["mean"] < 0))}
    return {
        "seed_count": 1,
        "seeds": {version: [11] for version in VERSIONS},
        "cross_seed_dispersion": None,
        "note": ("Single-seed evidence. No seed-29 arm exists, so no across-seed mean or "
                 "dispersion can be reported and the exact RankIC gap cannot be separated "
                 "from seed noise. The sign of the candidate-minus-CSZ difference is "
                 "consistent across all three development months, both block lengths and "
                 "the style-residualized view, which supports 'neither candidate is better "
                 "than CSZ' but not a precise magnitude."),
        "per_candidate": variants,
        "development_days": 65,
        "decision_windows": list(DEV_MONTHS),
        "context_window": "july2023 (pre-training for all three versions; not used for the decision)"}


def recommendation(archive: dict, paired: dict) -> dict:
    decision = archive["decision"]
    verdicts = {name: decision["candidates"][name]["verdict"] for name in CANDIDATES}
    if any(verdict == "better_than_csz" for verdict in verdicts.values()):
        raise ValueError("a candidate is marked better than CSZ; the H recommendation changes")
    return {
        "question": "submit RAW, CENTER, or keep CSZ and spend the slot on other improvements",
        "recommendation": "keep_csz",
        "submit": BASELINE,
        "submit_label": "V2 60-day CSZ labels (frozen seed-11 weights, current submission candidate)",
        "do_not_submit": list(CANDIDATES),
        "spend_slot_on_other_improvements": True,
        "rationale": [
            "RAW is clearly worse than CSZ on RankIC (-0.0247 pooled, all three months "
            "negative) and degrades every guardrail (turnover +17%, long-short and worst-day "
            "worse), so it is not a submission candidate.",
            "CENTER closes most of the RAW gap but still loses to CSZ on RankIC (-0.0046 "
            "pooled, all three months negative, block-5 interval entirely below zero) and "
            "degrades the long-short and worst-day guardrails; it does not pass the "
            "pre-registered gate.",
            "Neither candidate meets the pre-registered seed-29 trigger, so no additional "
            "training quota was spent; keeping V2-CSZ leaves the slot available for another "
            "direction instead of a candidate that already lost on identical keys and labels."],
        "followup_hint": ("If the CENTER-to-CSZ gap is worth chasing, test the CSZ-specific "
                          "variance/RMS matching as a single-factor experiment rather than "
                          "adding seeds to a candidate that already lost; the July context "
                          "window shows centering alone removes most of the RAW gap."),
        "verdicts": verdicts,
        "seed29_recommendation": decision["seed29_recommendation"]}


def build(path: Path) -> dict:
    archive = load_archive(path)
    before = {key: digest(value) for key, value in archive.items()}
    previous_sha = file_sha256(path)
    previous_committed_sha = committed_sha256(path)
    preregistration = verify_preregistration(archive)
    replay = verify_replay(archive)
    weights = verify_weights(archive)
    seed_coverage = verify_seed_coverage(archive)
    paired = paired_summary(archive)
    consolidation = {
        "schema_version": 1,
        "stage": "G5_final_consolidation",
        "created_at_utc": utc_now(),
        "scope": ("single-seed consolidation of the label-contrast round; no training, no "
                  "re-scoring, no official score"),
        "seed_coverage": seed_coverage,
        "preregistration": preregistration,
        "replay": replay,
        "weights": weights,
        "complete_table": complete_table(archive),
        "development_months_table": development_month_table(archive),
        "eight_time_point_summary": eight_time_point_summary(archive),
        "style_before_after": style_before_after(archive),
        "paired_summary": paired,
        "training_losses": archive["training_losses"]["per_arm"],
        "training_loss_note": archive["training_losses"]["note"],
        "decision": {"candidates": {name: {"verdict": archive["decision"]["candidates"][name]["verdict"],
                                           "seed29_followup": archive["decision"]["candidates"][name][
                                               "seed29_followup"],
                                           "checks": archive["decision"]["candidates"][name]["checks"]}
                                    for name in CANDIDATES},
                     "seed29_recommendation": archive["decision"]["seed29_recommendation"]},
        "evidence_strength": evidence_strength(archive, paired),
        "recommendation": recommendation(archive, paired),
        "consolidation_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": file_sha256(Path(__file__).resolve())},
        "provenance": {
            "supersedes_revision": 1,
            "previous_archive_sha256": previous_sha,
            "previous_committed_blob_sha256": previous_committed_sha,
            "previous_commit": "12b1145",
            "revision": 2,
            "note": ("Append-only revision: every revision-1 top-level block keeps the same "
                     "value digest; only this 'final_consolidation' block and the revision "
                     "metadata are added.")},
        "revision_history": [
            {"revision": 1, "stage": "G4_single_seed_evaluation", "commit": "12b1145",
             "archive_sha256": previous_sha,
             "committed_blob_sha256": previous_committed_sha},
            {"revision": 2, "stage": "G5_final_consolidation", "created_at_utc": utc_now(),
             "note": "single-seed consolidated report; seed 29 not triggered"}],
        "limitations": [
            "Only seed 11 exists for every arm; no across-seed mean or dispersion is available.",
            "Local approximate evaluation (jinyiquant.evaluation) and the frozen 2026-10-18 "
            "style-residualization convention; not the official platform score.",
            "Long-short returns are gross forward label returns with overlapping 30-minute "
            "horizons; block bootstrap describes day-level sampling noise only.",
            "july2023 is context only; the three development months were the pre-registered "
            "decision sample and are not an independent confirmation set."]}
    updated = dict(archive)
    updated["final_consolidation"] = consolidation
    updated["revision"] = 2
    after = {key: digest(value) for key, value in updated.items() if key in before}
    if after != before:
        raise ValueError("a revision-1 top-level block changed; refusing to write")
    write_json(path, updated)
    return updated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("archive", type=Path,
                              default=ROOT / "modified_tlob" / "label_contrast_20261024.json",
                              nargs="?")
    args = parser.parse_args(argv)
    updated = build(args.archive.resolve())
    consolidation = updated["final_consolidation"]
    print(args.archive.resolve())
    print("seed coverage", consolidation["seed_coverage"]["seeds"],
          "seed29_trained", consolidation["seed_coverage"]["seed29_trained"])
    for name, entry in consolidation["paired_summary"].items():
        raw = entry["raw_full"]["block5"]["rank_ic"]
        print(f"{name}: pooled rank_ic delta {raw['mean']:.5f} "
              f"[{raw['ci_low']:.5f}, {raw['ci_high']:.5f}]",
              "verdict", consolidation["decision"]["candidates"][name]["verdict"])
    print("recommendation", consolidation["recommendation"]["recommendation"],
          "submit", consolidation["recommendation"]["submit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
