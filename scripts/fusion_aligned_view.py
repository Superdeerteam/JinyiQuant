"""按标签键集对齐回放窗口，产出一份"只在有标签的键上成立"的只读视图。

背景：回放目录里 keys.parquet 覆盖全部预测键，而 targets.parquet 少最后时点没有标签的那几行
（十个窗口合计 1344 行），闸门的 read_window 要求两个文件键集一致。evaluate 本来也会在 merge
targets 时把这些行丢掉，所以对齐视图不改变评价对象；它只是把这个"只在有标签的行上评价"的事实
提前落到数据侧，好让闸门能读到。

做法：对每个窗口
  1. 以 targets 的 (date, instrument) 键集为准；
  2. 逐版本把 predictions 过滤到该键集，断言行数等于 targets 行数且无 NaN；
  3. 写 targets.parquet 与被过滤的 predictions，不写 keys.parquet（闸门在缺 keys 时用 targets 的键集）；
  4. 记录每个窗口每个版本被丢掉的行数与占比。

不动注册文件、不动闸门脚本；确认窗（jun2024/sep2024）默认不读。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

KEYS = ["date", "instrument"]
DEFAULT_CONFIRMATION = ("jun2024", "sep2024")


def align_window(run_dir: Path, version: str, window: str, support: pd.DataFrame,
                 expected: int, out_dir: Path) -> int:
    path = run_dir / "windows" / window / version / "predictions.parquet"
    if not path.is_file():
        raise SystemExit(f"missing predictions: {path}")
    predictions = pd.read_parquet(path)
    missing = [column for column in KEYS + ["score"] if column not in predictions.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    aligned = predictions.merge(support, on=KEYS, how="inner", validate="one_to_one")
    if len(aligned) != expected:
        raise SystemExit(f"{path}: {len(aligned)} aligned rows != targets rows {expected}")
    if aligned["score"].isna().any():
        raise SystemExit(f"{path}: NaN scores after alignment")
    target = out_dir / "windows" / window / version / "predictions.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    aligned[KEYS + ["score"]].sort_values(KEYS).reset_index(drop=True).to_parquet(
        target, index=False)
    return int(len(predictions) - len(aligned))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk-run", required=True, help="回放目录（含 windows/<win>/...）")
    parser.add_argument("--trunk-versions", required=True, help="逗号分隔，如 r441_s11,r441_s29")
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--windows", required=True, help="逗号分隔的窗口名")
    parser.add_argument("--out", required=True, help="对齐视图输出目录")
    parser.add_argument("--report", required=True, help="丢弃计数报告 json 路径")
    parser.add_argument("--reason", default="", help="写进报告的原因说明")
    args = parser.parse_args(argv)

    trunk_run = Path(args.trunk_run)
    candidate_run = Path(args.candidate_run)
    out_dir = Path(args.out)
    windows = [name.strip() for name in args.windows.split(",") if name.strip()]
    forbidden = [name for name in windows if name in DEFAULT_CONFIRMATION]
    if forbidden:
        raise SystemExit(f"refusing to open confirmation windows {forbidden}")
    trunk_versions = [name.strip() for name in args.trunk_versions.split(",") if name.strip()]

    per_window = []
    for window in windows:
        source = trunk_run / "windows" / window
        targets_path = source / "targets.parquet"
        if not targets_path.is_file():
            raise SystemExit(f"missing targets: {targets_path}")
        targets = pd.read_parquet(targets_path)
        support = targets[KEYS].sort_values(KEYS).reset_index(drop=True)
        expected = len(support)
        keys_path = source / "keys.parquet"
        keys_rows = len(pd.read_parquet(keys_path)) if keys_path.is_file() else expected
        entry = {
            "window": window,
            "keys_rows": int(keys_rows),
            "targets_rows": int(expected),
            "rows_dropped": int(keys_rows - expected),
            "versions": {},
        }
        if entry["rows_dropped"] < 0:
            raise SystemExit(f"{window}: targets longer than keys ({expected} > {keys_rows})")
        for version in trunk_versions:
            entry["versions"][version] = {
                "source": str(trunk_run / "windows" / window / version / "predictions.parquet"),
                "rows_dropped": align_window(trunk_run, version, window, support, expected, out_dir),
            }
        entry["versions"][args.candidate_version] = {
            "source": str(candidate_run / "windows" / window / args.candidate_version
                          / "predictions.parquet"),
            "rows_dropped": align_window(candidate_run, args.candidate_version, window, support,
                                         expected, out_dir),
        }
        target_path = out_dir / "windows" / window / "targets.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(targets_path.read_bytes())
        per_window.append(entry)

    totals = {
        "windows": len(per_window),
        "targets_rows": sum(row["targets_rows"] for row in per_window),
        "rows_dropped": sum(row["rows_dropped"] for row in per_window),
    }
    report = {
        "schema_version": "fusion-aligned-view/1",
        "purpose_zh": (
            "把回放窗口对齐到“有标签的键”上，使闸门 read_window 的键集检查能通过；"
            "evaluate 本来也会在 merge targets 时丢掉这些行，所以评价对象不变。"
        ),
        "reason": args.reason,
        "trunk_run": str(trunk_run),
        "candidate_run": str(candidate_run),
        "trunk_versions": trunk_versions,
        "candidate_version": args.candidate_version,
        "windows": per_window,
        "totals": totals,
        "keys_file_policy": (
            "视图里不写 keys.parquet：闸门在缺 keys 时用 targets 的键集（read_window 的既有回退），"
            "这样键集只有一个真源。"
        ),
        "consequences_zh": (
            "① 主指标不受影响：这些行没有标签，原本也不会进入 rank_ic/多空/换手的分母。"
            "② 唯一差异在平滑层：cross-sectional z-score 的分母少了这些行，受影响 timestamp 上"
            "的平滑分数会有极小差别；本报告记录每窗每版本的丢弃行数，供复核。"
            "③ 确认窗 jun2024/sep2024 未被读取（脚本硬拒绝）。"
        ),
        "confirmation_windows_touched": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "report": str(report_path), **totals},
                     ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
