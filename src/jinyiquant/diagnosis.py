"""Evidence-led hypotheses from verified learning audits; no model selection."""

import time
from pathlib import Path

import numpy as np

from .checks import verify_artifacts
from .io import file_sha256, read_json, utc_now, write_json
from .tracking import Recorder, capture_source, environment, finish

RULES = (
    ("small_sample_fit", "少量训练样本也拟合不好", "检查输入、目标尺度或优化过程；每轮只验证其中一个因素"),
    ("generalization", "训练好、后续日期明显差", "分别验证复杂度、正则化或提前停止"),
    ("date_concentration", "表现集中在少数几天", "在更多互不重叠日期复验当前方案"),
    ("history", "开盘或历史不足样本明显差", "分别验证历史窗口或缺失处理"),
    ("prediction_time", "不同预测时点差异明显", "固定时点评价，分别验证信息组合"),
    ("simple_model", "简单模型与 TLOB 相当", "检验结构必要性或输入信息是否充分"),
    ("reversals", "已有稳定信号，但频繁反转", "验证单一稳定性约束能否减少无效变化"),
    ("trade_information", "三档盘口有稳定贡献", "验证原始成交量价信息是否互补"),
)


def learning_findings(audit):
    """Treat missing/contradictory evidence as unknown, never as a diagnosis."""
    findings = audit.get("findings", {})
    reproduced = findings.get("same_seed_reproduction_passed") is True
    tiny = findings.get("small_sample_learning_passed")
    ratios = [row.get("train_mse_to_constant_ratio") for row in audit.get("runs", [])
              if row.get("arm") == "correct"]
    finite = bool(ratios) and all(isinstance(v, (float, int)) and not isinstance(v, bool)
                                 and np.isfinite(v) and v >= 0 for v in ratios)
    underfit = bool(finite and all(v > 1 for v in ratios))
    rules = [{"id": key, "finding": finding, "next_hypothesis": hypothesis, "status": "证据不足"}
             for key, finding, hypothesis in RULES]
    next_step = {"reason": "现有诊断未确定优先改动因素，先补充分日期、时点、历史覆盖和简单模型对照",
                 "factor": None, "hypothesis": None}
    if audit.get("passed") is True and reproduced:
        if tiny is False:
            rules[0]["status"] = "已观察到"
            next_step = {"reason": "小样本拟合未通过，优先排查学习过程",
                         "factor": "model.learning_rate",
                         "hypothesis": "只改变学习率，检验优化是否限制小样本拟合；输入与目标保持固定"}
        elif tiny is True:
            rules[0]["status"] = "未观察到"
            if underfit:
                next_step = {"reason": "小样本能拟合，但当前预算下全部正确标签训练均差于拟合常数",
                             "factor": "model.learning_rate",
                             "hypothesis": "只改变学习率，检验当前预算下的优化是否改善训练拟合及样本外表现"}
    return {"rules": rules, "training_fit": {"correct_run_count": len(ratios),
                "mse_to_constant_ratios": ratios if finite else [],
                "all_worse_than_constant": underfit if finite and reproduced and audit.get("passed") is True else None},
            "next_step": next_step, "trade_information_priority": "尚无稳定贡献证据，不设为首选",
            "result": "证据不足", "model_promoted": False,
            "limits": "诊断用于提出下一假设，不能证明原因或改善；新实验仍须预登记、校准与复验"}


def diagnose(audit_path, output_root):
    from .tracking import new_run_dir

    audit_path = Path(audit_path).resolve()
    destination = new_run_dir(Path(output_root).resolve())
    rec = Recorder(destination, "evidence-led-diagnosis")
    started = time.monotonic()
    write_json(destination / "manifest.json", {"schema_version": 1, "run_id": destination.name,
        "experiment_id": rec.experiment_id, "status": "running", "started_at": utc_now(),
        "code": capture_source(destination), "environment": environment()})
    rec.event("run", "started", learning_audit=str(audit_path), raw_data_reopened=False)
    try:
        manifest = read_json(audit_path / "manifest.json")
        if manifest.get("status") != "completed" or not verify_artifacts(audit_path)["passed"]:
            raise ValueError("Diagnosis requires a completed learning audit with intact artifacts")
        audit = read_json(audit_path / "metrics.json")
        if audit.get("passed") is not True:
            raise ValueError("Learning audit checks did not pass")
        result = learning_findings(audit)
        result.update(schema_version=1, evidence={"path": str(audit_path),
            "metrics_sha256": file_sha256(audit_path / "metrics.json"), "raw_data_reopened": False},
            evaluation_version=audit.get("evaluation_version"),
            metric_source="archived learning-audit metrics; no metric is recomputed here")
        write_json(destination / "diagnosis.json", result)
        write_json(destination / "checks.json", {"passed": True, "checks": [
            {"name": "intact_completed_learning_audit", "passed": True}]})
        rec.event("evaluation", "completed", result=result["result"], next_step=result["next_step"])
        finish(destination, "completed", 0, resources={"wall_seconds": time.monotonic() - started})
        return destination, 0
    except BaseException as exc:
        rec.failure("diagnosis", exc)
        finish(destination, "failed", 1, error=str(exc))
        raise
