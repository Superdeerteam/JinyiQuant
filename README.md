# JinyiQuant 研究工程入口

新增可选 [modified_tlob 回归模型](modified_tlob/README.md)：固定上游双轴主干、官方收益标签与 MSE，复用统一运行档案。该后端使用独立 Python 3.11/PyTorch 2.5.0 环境；原 Python 3.13 Ridge/Mean 工作流保留。安装、CPU 验收和平台提交包清单见该模块说明。

TLOB 本地训练入口为 `python -m modified_tlob.train <配置>`，本地只推理入口为 `python -m modified_tlob.infer --weights ... --bars ... --keys ... --output-root ...`；提交材料用 `python -m modified_tlob.package <训练运行目录> <新提交目录>` 生成。完整可复制命令、CPU/GPU 选择和输入输出说明见模块 README 的对应章节。

七个模块已统一为可复跑的本地 CPU 工作流。本轮以工程正确性验收；内置岭回归仅用于贯通接口。生成数据上的指标不能作为模型有效性或比赛成绩证据。

## 安装与首次运行

需要 Python 3.13 和 uv；`uv.lock` 锁定完整依赖。PowerShell 在仓库根目录执行：

```powershell
uv sync --locked
uv run jinyi demo --output data/demo
uv run jinyi validate-config data/demo/config.json
uv run jinyi run data/demo/config.json
```

命令输出 JSON，包括运行目录和退出码。目录每次唯一，配置中的相对路径按配置文件所在目录解析。查看输出的目录后，可以使用同一入口复验和检查：

```powershell
uv run jinyi rerun <运行目录>
uv run jinyi check <运行目录>
```

`validate-config` 默认只验证配置结构。增加 `--check-inputs` 后，只读检查已配置的行情、答案、股票池与映射路径是否为现有文件或目录，输出 `valid`（结构）、`inputs_present` 和逐项 `checks`；任一输入缺失时退出码为2。它不读取数据内容，不验证字段、覆盖率、模型依赖或GPU，也不代表数据质量与模型效果通过验收。

源码或输入数据变化后，复验会明确失败。要重现历史版本，在新的工作目录解压该次 `source.zip`，执行 `uv sync --locked`，配合该次 `config.json` 和校验一致的数据运行。模型和预处理参数一起保存，预测接口只接受历史窗口、股票和时点。

## 七个模块与产物

| 模块 | 实现 | 主要产物 |
|---|---|---|
| 数据接口 | `data.py`、`preparation.py` | 标准 `Batch`、`data.json`、`coverage.json`、`selection.json` |
| 配置管理 | `config.py` | 完整默认值与绝对路径的 `config.json`、`config.sha256` |
| 评价接口 | `evaluation.py` | `metrics.json`、`daily.csv`、原始与评分样本预测 |
| 实验记录 | `tracking.py` | `manifest.json`、`events.jsonl`、`source.zip`、`report.txt` |
| 统一入口 | `cli.py`、`runner.py` | 独立预算监控、状态/退出码、产物 SHA256 清单 |
| 自动验收 | `checks.py`、行为测试 | `checks.json`，产物完整性检查 |
| 复验与对照 | `experiments.py`、`comparison.py` | `calibration.json`、`comparison.json`、`decision.json` |

成功运行还保存 `model.npz`、`predictions.parquet`、`scored_predictions.parquet`、`submission.parquet`、`targets.parquet`。`submission.parquet` 的列是 `date/instrument/factor`，仅为本地接口产物；尚未通过线上平台提交流程。

退出码：0=执行完成（比较的结论仍需查看 `decision.json`）；1=运行失败；2=配置/调用错误；3=复验或产物检查失败；124=预算超限；130=中断。失败不会伪装为成功，已有部分产物、堆栈和恢复说明会保留。闭式岭回归只有一次训练步骤，学习率不适用，记录为 null。

## 数据与时间边界

输入支持 Parquet、Feather/Arrow、CSV、含月度 Parquet 的 RAR。RAR 读取依赖系统 libarchive `tar`（Windows 环境已验证），读取所需月份；Parquet 进行列投影和日期过滤。大型行情 tar.gz 先定向准备单个成员：

```powershell
uv run jinyi prepare-data cpt_jyc_2026_e2e_bar1m.tar.gz --member cpt_jyc_2026_e2e_bar1m/202308.0.feather --output data/bars
```

每次准备生成独立目录、文件 SHA256 和来源记录，防止路径穿越；不会全量解压或修改原始压缩包。输入可以是单文件或含月度分片的目录。可选 `data.cache_dir` 缓存按角色、源文件 SHA256、字段、日期与读取器版本隔离的投影数据；每次仍校验源与缓存内容，损坏缓存重建并写事件。缓存可重建，不属于复现必需产物。

键为 `date/instrument`，`date` 使用上海交易时间、不带时区，行情时间是 bar 结束时刻。下载版 `instrument_id` 必须提供一对一 `instrument_map_path`，不得从证券数字代码猜测。下载版价格和 amount 单位是分，设置 `price_scale=100`；线上元数据设置 1。映射必须来自同时含两列的官方表，示例 SQL 在 `configs/export_instrument_map.sql`。

标签读取官方 `vwap_return`（也接受标准 `target`），不自行替换成普通未来收益。缺失特征有显式掩码，按单只股票做有限长度前向填充或补零；缺失答案不填补，单独报告覆盖。预测股票来自当时股票池，答案仅在评价阶段对齐；未提供股票池时从当时实际行情键推导，无法证明原始数据外的股票/整个截面是否缺失，正式验收应提供官方股票池。

个别数据问题默认记录后继续：`data.outside_universe_labels="exclude"` 按当时股票池排除池外答案，在 `data.json` 和 WARNING 事件记录数量；需要严格排查时设为 `error`。股票池中缺行情的股票仍保留预测键，完全无历史时返回训练期目标均值，避免把补零当成真实价格而产生极端分数；训练时排除未达到 `min_history` 的样本。`coverage.json`、`metrics.json` 和 `daily.csv` 报告缺失答案、缺历史及实际评分样本数。

是否继续由**每个截面**的历史覆盖率和答案覆盖率决定，训练/验证都用独立股票池作分母。低于 `acceptance.min_coverage` 或整个截面不可用会失败，重复/歧义映射、时间越界、非有限预测和标签泄漏仍会失败。默认覆盖阈值0.6对应资料中的最多40%缺失上限；真实工程小切片另声明0.99，最多容忍1%缺失，不因结果事后降低。极少数缺失记录的通过不代表全量质量或模型收益已验收。

当前已解压的真实数据位于 `数据/`。两份真实示例配置直接使用行情月度目录、标签月度目录、股票池 `cpt_jyc_2026_instruments(1).parquet` 和映射CSV，无需再次解压。运行前先检查路径：

```powershell
uv run jinyi validate-config configs/real_smoke.json --check-inputs
uv run jinyi run configs/real_smoke.json
uv run jinyi rerun <上一步输出的运行目录>
```

该配置训练2023-08-01、验证2023-08-02，最终区间不读取。历史窗口最多30条，允许开盘不足完整窗口的样本（至少1条已观测记录），这属于明确的工程演示配置。输入路径检查通过不代表真实训练、数据质量或预测效果已经验收；TLOB训练仍需使用其独立Python3.11环境。

训练、研究验证、最终检查采用固定的半开区间，标签窗口必须完全处于区间内。默认开发许可区间为 2020-01-01 至 2025-01-01，股票池中更晚的日期不会自动纳入。历史窗口只用预测时点之前或当时的已结束 bar，跨日使用历史观测行；`lookback` 表示观测条数，不保证连续自然分钟。缺省八个采样时点来自本地股票池实测。11:30 标签按交易时间假定到13:30，这一午休假设在 `selection.json` 标注，取得更明确官方口径后应复核。

默认只评价 validation。评价 final 必须配置 `evaluation_split=final`、非空 `final_access.purpose` 和查看上限，SQLite 账本记录访问次数，改实验名称不能绕过同数据/同区间的限制；对照模块拒绝用 final 结果选模。此账本是本地治理措施，不是防止用户更改文件或配置的安全边界。

## 公平对照

先对固定基线执行多种子、互不重叠时段的波动测量：

```powershell
uv run jinyi calibrate data/demo/config.json --seeds 11 29 --periods configs/demo_periods.json
```

再准备 suite JSON，字段为 `baseline_config`、`candidate_config`、`seeds`、`periods`、`calibration_report`、`policy`。路径相对于 suite 文件。阈值要依据基线波动与实际需求事先声明，工具不擅自设定“提升5%”。policy 拒绝未知字段，必填项如下：

| 字段 | 约定 |
|---|---|
| `primary_metric`、`direction`、`min_improvement` | 已实现的按日指标、maximize/minimize、事先声明的实际改善量 |
| `max_metric_regressions` | 非空次要指标字典，值为允许退化量或 `{max_regression, direction}` |
| `max_wall_ratio`、`max_memory_ratio` | 每个配对的候选/基线资源比值上限 |
| `min_seeds`、`min_periods` | 均至少2；时段互不重叠 |
| `bootstrap_samples`、`block_days`、`confidence`、`seed` | 至少100次连续日块抽样、块长度、置信度、抽样种子 |
| `threshold_rationale` | 解释基线波动、实际改善门槛与成本限额 |

主指标或次要指标涉及 RankIC/IC 时，对照必须覆盖全部由标签决定的可评价截面，并同时核对截面数和完整时间戳的 SHA256。标签为常数的截面不纳入相关评价；候选在可评价截面输出常数导致覆盖不足时拒绝，基线覆盖不足或旧档案缺少覆盖证据时结论为 `inconclusive`。描述指标仍保留未定义相关性，不能用忽略这些值后的均值证明改善。

```powershell
uv run jinyi suite <suite.json>
```

Suite 先冻结策略、假设和预算，再为每个种子/时段分别执行基线、基线复验、候选、候选复验；单次失败即停止且保留全部尝试。比较数据、时间切分、预算、样本、答案、源码、评价口径及成本；通过真实复跑产物验证容差。按交易日配对连续日块 bootstrap，同一时段跨种子使用相同抽样，不把同日股票当独立样本。

只有多种子/多时段复验、统计及实际门槛、次要指标和成本检查均通过，decision 才建议保留候选。证据不足为 `inconclusive`，不满足要求为 `reject`。所有历史基线保留，不自动替换模型。多次筛选仍有选择偏差，需新的验证证据；此工具不声称自动消除了选择偏差。

## 自动验收与证据

```powershell
uv run ruff check .
New-Item -ItemType Directory -Force scratch | Out-Null
uv run pytest -q --basetemp scratch/tests-001
uv run python scripts/acceptance.py --output runs/engineering-acceptance
```

验收脚本从头生成固定种子数据，执行正常运行、相同配置复验、超时、超内存、缺文件、无效配置脱敏、基线波动测量及多种子/多时段对照。候选故意与基线相同，必须得出“证据不足”，用于检查系统不会把零改善误判为成功。脚本不覆盖已有输出；重复运行须换新目录。当前系统临时目录权限有限，因此测试明确使用工作区内的新临时目录。

`audit.sqlite3` 是事务化运行/最终查看索引，`index.jsonl` 为便利索引。每次运行的日志、配置、源码、指标、检查和模型持久保留；无自动删除历史/最佳版本策略。大数据压缩包、派生缓存和 runs 不纳入 Git，保留源数据校验信息；需要长期异地归档时应整体复制验收通过的运行目录及匹配数据。Ridge/Mean 不使用 GPU；modified_tlob 已完成 CPU 神经网络工程闭环，GPU 实际计算仍未验收，详细证据见模块说明。

具体本地指标口径保存在每次运行的 `metrics.json/conventions`。公开实现核对参考 [SciPy 的常数输入相关性约定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)与 [NumPy 的缺失值均值说明](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html)；对照额外要求固定的标签可评价截面集合。工程记录保留在独立运行档案中。

## 仓库发布边界

仓库只包含代码、测试、配置模板、依赖锁和说明，不上传行情、股票池、答案、证券映射CSV、缓存、模型或运行档案。`.gitignore` 与 `scripts/check_repository_payload.py` 共同检查；提交前执行 `uv run python scripts/check_repository_payload.py --mode staged`，`--mode tracked` 检查完整仓库文件清单。GitHub Actions 在 Windows/Linux 检查依赖、静态规则、行为测试和仓库文件清单。
