# 第一轮接口与验收约定

目标：实现本地、CPU、可复跑的研究基础设施。闭式岭回归仅用于验证工程链路，不代表研究方向或有效交易模型。原始数据不修改，历史运行不覆盖，默认不访问最终检查区间。

验收：严格配置；键与数值校验；按股票因果历史窗口；30分钟标签在切分内完成；推理接口不接收答案；未来行情扰动不影响此前预测；训练归一化不接触验证数据；正常/失败/预算退出均记录；相同种子重跑按预先容差比对；对照必须相同数据、切分、评价和预算，按日配对连续区块 bootstrap，不因单次胜出晋升。

## 模块约定（内部接口）

- `config.py`: Pydantic frozen `RunConfig`。`data` 包含 `bars_path`, `labels_path`, `instruments_path`、`instrument_map_path`、`cache_dir` (后三者可选), `feature_columns`, `lookback`, `history_days`, `missing_policy` (`zero`/`ffill`), `max_ffill_steps`, `min_history`, `label_horizon_minutes=30`, `sample_times` (HH:MM), `price_scale`, `missing_sentinel`, `source`, `version`, `allowed_start/end`。时间列统一 `date`，证券列统一 `instrument`，标签列 `target`（输入允许 `vwap_return`），标签结束列 `label_end`（缺省按30交易分钟推断，11:30→13:30的午休假设明确记录；不跨日）。`splits.train/validation/final` 是 start/end 半开区间，`evaluation_split` 默认 validation。路径在 load_config 时对配置文件目录解析。`model` 有 `kind='ridge'|'mean'`, `alpha`；`seed`；`budget` 有 wall_seconds,max_rss_mb,max_train_samples；`evaluation` 有 quantile,periods_per_year（按日默认252）,min_cross_section；`acceptance` 有 min_coverage,repro_atol,repro_rtol；`logging.resource_interval_seconds`。
- `data.py`: `load_data(config.data, start=None, end=None) -> DataBundle`：bars/labels/universe pandas DataFrame、metadata 字典、cache_events事件。只返回所需开发日期，自动回溯history_days并裁剪到许可起点；metadata含数据文件sha256和数据版本。`build_batch(bars, keys, data_config) -> Batch`：keys(date,instrument)、x[n,lookback,n_features]、observed[同形bool]、history_dates[n,lookback]，coverage字典；无标签。`select_targets(labels, split, horizon_minutes)` 返回标签结束在该半开区间内的标签表。`Batch`属性 keys,x,observed,history_dates,coverage。键按 date,instrument 稳定排序。输入缺失通过 mask+因果填补，标签不填补，重复/非法数据明确报错。推理键独立生成；只有评价阶段显式选择答案有效的样本，并检查覆盖率。
- `evaluation.py`: `evaluate(predictions, targets, evaluation_config) -> (metrics:dict,daily:DataFrame)`；predictions仅 date,instrument,score；完整严格键对齐，不静默丢弃。按截面RankIC/PearsonIC、分组多空、换手，日聚合与成本无关；标注local_approximation，不输出官方综合分。daily列 date,rank_ic,ic,long_short_return,turnover；metrics 至少 rank_ic_mean,rank_ic_ir,ic_mean,long_short_sharpe,turnover_mean,extreme_return,min_daily_long_short_return。
- `comparison.py`: `compare_runs(baseline_dirs, candidate_dirs, policy:dict) -> (comparison,decision)`。读取manifest/config/metrics/daily/checks，严格验证可比性与配对(seed, validation start/end)，区块bootstrap由numpy实现、固定种子，按日而非股票抽样，检查多个种子/时间段、重跑容差证据、成本、次要指标；缺证据为inconclusive。policy需声明 primary_metric,direction,min_improvement,max_metric_regressions,max_wall_ratio,max_memory_ratio,min_seeds,min_periods,bootstrap_samples,block_days,confidence,seed,threshold_rationale，不能默认凭空提升门槛。返回JSON可序列化，无NaN。
- `tracking.py`：运行档案、结构化事件、源代码快照、资源/预算。
- `models.py`：共享训练标准化与推理，fit仅训练批次，预测仅Batch。
- `checks.py`：自动行为检查。
- `runner.py`：串联全部模块；cli统一入口，预算进程监控，退出码和失败持久化。

## 数据差异的默认容忍规则

`data.outside_universe_labels` 默认为 `exclude`，按当时股票池显式排除池外答案并记录统计/警告，也可设 `error`。不删除股票池的预测键；完全缺历史的预测返回训练目标均值，训练仅使用达到min_history的观测样本。训练和验证均对独立股票池检查逐截面历史/答案覆盖率，至少达到acceptance.min_coverage；整段缺失、重复或歧义映射、未来数据依赖依然失败。统计进入data.json/alignment、coverage.json、metrics.json/coverage及daily.csv，不以补零或减少分母掩盖缺失。

产物：config.json/config.sha256、data.json、coverage.json、manifest.json、events.jsonl、model.npz、predictions.parquet、targets.parquet、metrics.json、daily.csv、checks.json、source.zip、report.txt。比较另有comparison.json/decision.json。运行目录不可覆盖，index.jsonl持久记录。最终检查有独立访问账本与查看上限，比较拒绝最终检查结果选模。
