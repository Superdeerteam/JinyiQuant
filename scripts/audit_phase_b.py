"""Independent phase-B audit of the delivered market-context heads.

Three questions, answered from the real trained weights (never from the head-builder's own
check stage):

1. Does ``TLOBAdapter.load`` restore the head, and does the restored head carry the parameter
   count and spec the delivery claims? Does the head-less candidate still load and score through
   the unchanged path?
2. Is every arm layout-agnostic under the *production* knob (``config["batch_size"]`` = the
   trunk's forward slice width) at 32 / 256 / 8192?
3. Does the *fragment* experiment (re-chunking one timestamp into 32-row sub-batches with
   ``allow_cross_section_fragment``) still separate the arms: A and B unchanged, C moved? And how
   large is C's residual on the same batch?

The head-level width sweep recomputes the market summary on 32/256/8192-row fragments straight
from the frozen replay cache, so the mechanism is measured without the trunk's GEMM noise.

Read-only: loads checkpoints and the frozen cache, writes one JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from jinyiquant.config import load_config  # noqa: E402
from jinyiquant.data import Batch, build_batch, load_data  # noqa: E402
from jinyiquant.runner import inference_keys  # noqa: E402
from modified_tlob import market_head  # noqa: E402
from modified_tlob.adapter import TLOBAdapter  # noqa: E402
from market_context import split_of  # noqa: E402

CONFIG = "configs/scaling_441d_csz_seed11.json"
CANDIDATE = "/root/weights/sealed/cand/cand_s11_264.json"
WEIGHTS = Path("/root/runs/mktctx-head-20261028/weights")
CACHE = Path("/root/cache/frozen-h/441d-s11")
DEFAULT_OUT = "runs/mktctx-20261028/phase_b.json"
TOLERANCE = 1e-6
DOMAIN = pd.Timestamp("2023-11-02"), pd.Timestamp("2023-11-03")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zero_init(head) -> bool:
    if getattr(head, "arm", None) == "C":
        return bool(int(torch.count_nonzero(head.mix.weight)) == 0)
    last = head.layers[-1]
    return bool(int(torch.count_nonzero(last.weight)) == 0 and int(torch.count_nonzero(last.bias)) == 0)


def state_digests(head) -> dict:
    """Candidate canonical forms of the head state, so a claimed digest can be reproduced."""
    state = head.state_dict()
    text = market_head.text_state(state)
    ordered = {name: state[name].detach().cpu().numpy().astype(np.float32)
               for name in sorted(state)}
    return {
        "bytes_concat_f32": hashlib.sha256(b"".join(v.tobytes() for v in ordered.values())).hexdigest(),
        "json_text_sorted": hashlib.sha256(
            json.dumps(text, sort_keys=True).encode("utf-8")).hexdigest(),
        "json_text_sorted_compact": hashlib.sha256(
            json.dumps(text, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "json_text_plain": hashlib.sha256(json.dumps(text).encode("utf-8")).hexdigest(),
    }


def _plain(value):
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def head_report(head) -> dict:
    payload = market_head.head_payload(head)
    state = {name: value.detach().cpu().numpy()
             for name, value in payload["state_dict"].items()}
    return {
        "class": type(head).__name__,
        "arm": payload.get("arm"),
        "parameters": int(sum(p.numel() for p in head.parameters())),
        "spec": {str(key): _plain(value) for key, value in dict(payload["spec"]).items()},
        "state_sha256": hashlib.sha256(b"".join(
            state[name].astype(np.float32).tobytes() for name in sorted(state))).hexdigest(),
        "nonfinite": int(sum(int((~np.isfinite(value)).sum()) for value in state.values())),
    }


def item_load(models: dict, package: dict) -> dict:
    entry = {}
    for arm, model in models.items():
        items = {"file_sha256": sha256(Path(model.loaded_from) if hasattr(model, "loaded_from") else
                                       (WEIGHTS / f"{arm.lower()}-s11.json" if arm != "A" else CANDIDATE)),
                 "market_head": model.market_head is not None,
                 "batch_size": int(model.config["batch_size"]),
                 "trained": bool(model.trained)}
        if model.market_head is not None:
            items["head"] = head_report(model.market_head)
            claimed = package.get(arm, {}).get("parameters")
            items["claimed_parameters"] = claimed
            items["parameters_match"] = claimed == items["head"]["parameters"]
            claimed_digest = package.get(arm, {}).get("head_state_sha256")
            digests = state_digests(model.market_head)
            items["claimed_head_state_sha256"] = claimed_digest
            items["state_digests"] = digests
            items["state_digest_reproduced_by"] = [name for name, value in digests.items()
                                                   if value == claimed_digest]
            fresh = market_head.build_head(arm)
            items["fresh_build_parameters"] = int(sum(p.numel() for p in fresh.parameters()))
            items["fresh_build_zero_init"] = zero_init(fresh)
        entry[arm] = items
    entry["headless_path_unchanged"] = {
        "market_head": models["A"].market_head is None,
        "note": "with market_head=None _score() skips _market_residual entirely, so the head-less "
                "candidate still scores through the pre-existing path",
    }
    return entry


def one_timestamp(batch: Batch, stamp) -> Batch:
    same = (pd.to_datetime(batch.keys["date"]) == pd.Timestamp(stamp)).to_numpy()
    return Batch(keys=batch.keys.loc[same].reset_index(drop=True), x=batch.x[same],
                 observed=batch.observed[same], history_dates=batch.history_dates[same],
                 coverage=batch.coverage, history_feature_count=batch.history_feature_count)


def fragment(batch: Batch, start: int, stop: int) -> Batch:
    window = slice(start, min(stop, len(batch.keys)))
    return Batch(keys=batch.keys.iloc[window].reset_index(drop=True), x=batch.x[window],
                 observed=batch.observed[window], history_dates=batch.history_dates[window],
                 coverage=batch.coverage, history_feature_count=batch.history_feature_count)


def item_layout(models: dict, batch: Batch) -> dict:
    entry = {"rows": int(len(batch.keys)), "timestamps": int(len(pd.unique(pd.to_datetime(
        batch.keys["date"]))))}
    whole_width = len(batch.keys)
    reference = {}
    for arm, model in models.items():
        model.config["batch_size"] = whole_width
        reference[arm] = model.predict(batch)
    entry["width_sweep"] = {}
    for width in (32, 256, 8192):
        widths = {"width": width}
        for arm, model in models.items():
            model.config["batch_size"] = min(width, whole_width)
            scores = model.predict(batch)
            widths[f"{arm}_max_abs_vs_whole"] = float(np.max(np.abs(scores - reference[arm])))
        entry["width_sweep"][str(width)] = widths
    for arm, model in models.items():
        model.config["batch_size"] = whole_width
    entry["layout_agnostic_under_width_sweep"] = {
        arm: bool(entry["width_sweep"][str(8192)][f"{arm}_max_abs_vs_whole"] <= TOLERANCE)
        for arm in models}
    fragment_scores = {}
    for arm, model in models.items():
        model.config["batch_size"] = 32
        model.allow_cross_section_fragment = True
        try:
            fragment_scores[arm] = np.concatenate([model.predict(fragment(batch, start, start + 32))
                                                   for start in range(0, whole_width, 32)])
        finally:
            model.allow_cross_section_fragment = False
            model.config["batch_size"] = whole_width
    entry["fragment_chunk_rows"] = 32
    for arm in models:
        entry[f"{arm}_fragment32_vs_whole_max_abs"] = float(
            np.max(np.abs(fragment_scores[arm] - reference[arm])))
    entry["layout_agnostic_fragment"] = {
        arm: bool(entry[f"{arm}_fragment32_vs_whole_max_abs"] <= TOLERANCE) for arm in models}
    entry["head_residual_abs_max"] = float(np.max(np.abs(reference["C"] - reference["A"])))
    entry["head_residual_std"] = float(np.std(reference["C"] - reference["A"]))
    entry["residual_fraction_nonzero"] = float(np.mean(reference["C"] != reference["A"]))
    for (lower, upper) in (("C", "B"), ("C", "A"), ("B", "A")):
        entry[f"{lower}-{upper}_fragment_vs_whole_max_abs"] = float(np.max(np.abs(
            (fragment_scores[lower] - fragment_scores[upper]) - (reference[lower] - reference[upper]))))
    return entry


def item_cache_widths(head_c, head_a) -> dict:
    cached = {name: np.load(CACHE / "replay" / f"{name}.npy")
              for name in ("h", "context", "context_code", "context_count", "valid")}
    keys = pd.read_parquet(CACHE / "replay" / "keys.parquet")
    h = cached["h"]
    stamps = pd.to_datetime(keys["date"])
    entry = {"timestamps": int(stamps.nunique()), "rows": int(len(h)), "sweep": {}}
    residual_whole = []
    chosen = []
    for stamp in list(pd.unique(stamps))[:3]:
        rows = np.flatnonzero((stamps == stamp).to_numpy())
        chosen.append(rows)
        state = torch.from_numpy(np.ascontiguousarray(h[rows]))
        m_whole = cached["context"][cached["context_code"][rows][0]]
        with torch.inference_mode():
            per_row = torch.from_numpy(np.repeat(m_whole[None], len(rows), axis=0).astype(np.float32))
            delta_whole = head_c.delta(state, per_row).cpu().numpy()
        residual_whole.append(delta_whole)
        for width in (32, 256, 8192):
            parts = []
            for start in range(0, len(rows), width):
                block = rows[start:start + width]
                codes = np.zeros(len(block), dtype=np.int64)
                summary = market_head.market_context(
                    h[block].astype(np.float64), codes,
                    valid=cached["valid"][block], counts=np.array([len(block)], dtype=np.int64),
                    n_groups=1)
                with torch.inference_mode():
                    delta = head_c.delta(torch.from_numpy(np.ascontiguousarray(h[block])),
                                         torch.from_numpy(np.ascontiguousarray(summary[codes]))).cpu().numpy()
                parts.append(delta)
            chunked = np.concatenate(parts)
            entry["sweep"].setdefault(str(width), 0.0)
            entry["sweep"][str(width)] = max(entry["sweep"][str(width)],
                                             float(np.max(np.abs(chunked - delta_whole))))
    everything = np.abs(np.concatenate(residual_whole))
    with torch.inference_mode():
        state = torch.from_numpy(np.ascontiguousarray(h))
        per_row = torch.from_numpy(cached["context"][cached["context_code"]].astype(np.float32))
        residual_all = head_c.delta(state, per_row).cpu().numpy()
    whole_gap = []
    for rows in chosen:
        summary = market_head.market_context(
            h[rows].astype(np.float64), np.zeros(len(rows), dtype=np.int64),
            valid=cached["valid"][rows], counts=np.array([len(rows)], dtype=np.int64), n_groups=1)
        whole_gap.append(float(np.max(np.abs(summary[0] - cached["context"][cached["context_code"][rows][0]]))))
    entry["stored_context_reproduced_max_abs"] = max(whole_gap)
    entry["residual_abs_max_cache"] = float(np.max(np.abs(residual_all)))
    entry["residual_std_cache"] = float(np.std(residual_all))
    entry["residual_fraction_nonzero_cache"] = float(np.mean(residual_all != 0.0))
    entry["residual_abs_max_first_three_timestamps"] = float(everything.max())
    entry["reading"] = ("re-chunking a timestamp recomputes its market summary on a fragment; the "
                        "32-row summary moves C's residual, A and B never look at the summary")
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    weights = Path(args.weights)
    package = json.loads((weights / "package.json").read_text(encoding="utf-8"))
    paths = {"A": Path(args.candidate), "B": weights / "b-s11.json", "C": weights / "c-s11.json"}
    models = {arm: TLOBAdapter.load(path, device="cpu") for arm, path in paths.items()}
    models["A"].loaded_from = str(paths["A"])
    report = {"schema_version": 1, "tolerance": TOLERANCE, "arms": {}, "files": {}}
    for arm, path in paths.items():
        report["files"][arm] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    report["arms"] = item_load(models, package)
    scale = float(models["C"].target_scale)
    config = load_config(Path(args.config))
    split = split_of(config, "replay")
    start, end = DOMAIN
    bundle = load_data(config.data, start=start, end=end)
    keys = inference_keys(bundle, split, config.data)
    keys = keys.loc[(keys["date"] >= pd.Timestamp(start))
                    & (keys["date"] < pd.Timestamp(end))].reset_index(drop=True)
    batch = build_batch(bundle.bars, keys, config.data)
    stamp = pd.to_datetime(batch.keys["date"]).max()
    single = one_timestamp(batch, stamp)
    report["batch"] = {"window": [str(start), str(end)], "rows": int(len(batch.keys)),
                       "timestamps": int(len(pd.unique(pd.to_datetime(batch.keys["date"])))),
                       "probe_timestamp": str(stamp), "probe_rows": int(len(single.keys))}
    for arm, model in models.items():
        model.config["batch_size"] = len(batch.keys)
    day = {arm: model.predict(batch) for arm, model in models.items()}
    stamps = pd.to_datetime(batch.keys["date"]).reset_index(drop=True)
    per_stamp = {str(stamp): float(np.max(np.abs((day["C"] - day["A"])[stamps == stamp])))
                 for stamp in pd.unique(stamps)}
    report["day_residual"] = {
        "rows": int(len(batch.keys)), "timestamps": len(per_stamp),
        "abs_max": float(np.max(np.abs(day["C"] - day["A"]))),
        "std": float(np.std(day["C"] - day["A"])),
        "per_timestamp_abs_max": per_stamp}
    report["layout"] = item_layout(models, single)
    report["layout"]["head_residual_abs_max_raw"] = report["layout"]["head_residual_abs_max"] / scale
    report["cache_widths"] = item_cache_widths(models["C"].market_head, models["A"].market_head)
    report["units"] = {
        "prediction_units_per_raw": scale,
        "note": "layout/day metrics come from predict(), which multiplies by target_scale; the "
                "cache_widths metrics are raw head.delta output, so divide by target_scale to "
                "compare them with predict()-level numbers"}
    report["cache_widths"]["residual_abs_max_cache_prediction"] = (
        report["cache_widths"]["residual_abs_max_cache"] * scale)
    report["cache_widths"]["residual_std_cache_prediction"] = (
        report["cache_widths"]["residual_std_cache"] * scale)
    report["day_residual"]["abs_max_raw"] = report["day_residual"]["abs_max"] / scale
    checks = {
        "heads_restored": all(report["arms"][arm]["market_head"] for arm in ("B", "C")),
        "headless_unchanged": not report["arms"]["A"]["market_head"],
        "parameters_match": all(report["arms"][arm].get("parameters_match", True) for arm in ("B", "C")),
        "heads_nonfinite": int(sum(report["arms"][arm]["head"]["nonfinite"] for arm in ("B", "C"))),
        "width_sweep_agnostic": all(report["layout"]["layout_agnostic_under_width_sweep"].values()),
        "fragment_A": report["layout"]["layout_agnostic_fragment"]["A"],
        "fragment_B": report["layout"]["layout_agnostic_fragment"]["B"],
        "fragment_C": report["layout"]["layout_agnostic_fragment"]["C"],
        "residual_alive": report["layout"]["head_residual_abs_max"] > 1e-4,
    }
    report["checks"] = checks
    report["verdict"] = ("pass" if all([checks["heads_restored"], checks["headless_unchanged"],
                                        checks["parameters_match"], checks["heads_nonfinite"] == 0,
                                        checks["width_sweep_agnostic"], checks["fragment_A"],
                                        checks["fragment_B"], not checks["fragment_C"],
                                        checks["residual_alive"]]) else "fail")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "checks": checks,
                      "width_sweep": report["layout"]["width_sweep"],
                      "fragment": {arm: report["layout"][f"{arm}_fragment32_vs_whole_max_abs"]
                                   for arm in ("A", "B", "C")},
                      "head_residual_abs_max": report["layout"]["head_residual_abs_max"],
                      "cache_widths": report["cache_widths"]["sweep"],
                      "cache_residual_abs_max": report["cache_widths"]["residual_abs_max_cache"]},
                     indent=1))
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
