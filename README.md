# JinyiQuant 研究工程入口

可选 [modified_tlob 回归模型](modified_tlob/README.md) 使用上游双轴主干、官方收益标签与 MSE，复用统一运行档案。该后端使用独立 Python 3.11 环境，CPU 基础锁为 PyTorch 2.5.0，当前 RTX 5060 研究环境为 2.7.1+cu128；原 Python 3.13 Ridge/Mean 工作流保留。安装、CPU/GPU 验收状态和平台提交包清单见模块说明。

TLOB 本地训练入口为 `python -m modified_tlob.train <配置>`，本地只推理入口为 `python -m modified_tlob.infer --weights ... --bars ... --keys ... --output-root ...`；提交材料用 `python -m modified_tlob.package <训练运行目录> <新提交目录>` 生成。完整可复制命令、CPU/GPU 选择和输入输出说明见模块 README 的对应章节。

七个模块已统一为可复跑的本地工作流。2023年7月真实数据已完成 CUDA 训练对照及 CPU/GPU 保存加载检查；官方平台、全区间 ≤3 小时和最终检查仍未验收。生成数据上的指标不能作为模型有效性或比赛成绩证据。

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

原真实示例配置使用 `数据/` 下的行情月度目录、标签月度目录、股票池 `cpt_jyc_2026_instruments(1).parquet` 和映射CSV；当前7月研究行情位于 `data/research-july-20261015/bars/`，对应来源与校验值随运行归档。运行示例前先检查路径：

```powershell
uv run jinyi validate-config configs/real_smoke.json --check-inputs
uv run jinyi run configs/real_smoke.json
uv run jinyi rerun <上一步输出的运行目录>
```

该配置训练2023-08-01、验证2023-08-02，最终区间不读取。历史窗口最多30条，允许开盘不足完整窗口的样本（至少1条已观测记录），这属于明确的工程演示配置。输入路径检查通过不代表真实训练、数据质量或预测效果已经验收；TLOB训练仍需使用其独立Python3.11环境。

训练、研究验证、最终检查采用固定的半开区间，标签窗口必须完全处于区间内。默认开发许可区间为 2020-01-01 至 2025-01-01，股票池中更晚的日期不会自动纳入。历史窗口只用预测时点之前或当时的已结束 bar，跨日使用历史观测行；`lookback` 表示观测条数，不保证连续自然分钟。缺省八个采样时点来自本地股票池实测。11:30 标签按交易时间假定到13:30，这一午休假设在 `selection.json` 标注，取得更明确官方口径后应复核。

默认只评价 validation。评价 final 必须配置 `evaluation_split=final`、非空 `final_access.purpose` 和查看上限，SQLite 账本记录访问次数，改实验名称不能绕过同数据/同区间的限制；对照模块拒绝用 final 结果选模。此账本是本地治理措施，不是防止用户更改文件或配置的安全边界。

## 公平对照

### 当前7月训练对照

使用独立 `.venv-tlob-cuda` 环境，依赖锁为 `modified_tlob/requirements-blackwell.lock.txt`（安装步骤及研究选项见模块 README）：

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-tlob-cuda/Scripts/python.exe scripts/run_training_plan.py configs/training_step1_plan.json
```

本轮五步筛选及按预定门槛触发的复验已完成。结果、保留/回退依据与运行档案见[结构化研究记录](modified_tlob/training_progression_20261015.json)和[最终研究报告](runs/training-progression/final-report-20261015/summary_zh.txt)；配对结果见[训练时点对照图](runs/training-progression/figures/step3c_training_times_paired.png)及[训练轮数对照图](runs/training-progression/figures/step1_epochs_validation_mse.png)。数据、模型与完整运行档案留在本地。

研究主候选为十个训练日、全部8时点、原始12列盘口、`target_scaling=train_std`、`time_loss_weighting=none`，不做信号平滑。本地参考固定使用最新研究验证窗（2023-07-25至07-28）的 seed 11，保存在[本地研究参考包](dist/local-research-reference-20261015-raw12-all8-s11-july25/)，并非正式冠军。

首轮2→12轮对照已完整执行；扩展数据后的十日/八时点实验固定1500次更新、batch_size=32，约为0.6个 epoch，不等于在扩展数据上完整训练12轮。证据仅覆盖8个研究日，仍有09:30开盘退化和反复筛选的局限；最终检查区间未打开，官方平台及全区间时限尚未验收。

当前配方的训练时长对照已完成：固定其余条件，比较1500/5000/10000次更新（约0.6/2/4轮），共六组成功训练。两种长预算均未通过预注册初筛，本轮保留1500步研究参考；10000步在第一窗口改善，但第二窗口排序、多空和换手退化，未触发额外种子复验。配置、失败恢复及保留依据见[训练时长研究记录](modified_tlob/training_duration_20261015.json)，详细结果见[本地报告](runs/training-duration-20261015/screen-analysis/summary.txt)与[学习曲线](runs/training-duration-20261015/screen-analysis/training_duration.png)。这一结论限于当前配方和研究窗口，不代表已精确定位全局最优训练步数。

### 真实后续日期三参照基线

先运行短测，再冻结正式计划。使用现有 TLOB 环境并安装可选绘图依赖：

```powershell
uv pip install --python .venv-tlob/Scripts/python.exe -r modified_tlob/requirements-research.lock.txt
$env:PYTHONPATH = 'src;.'
& .venv-tlob/Scripts/python.exe -m jinyiquant run configs/later_date_speed.json
& .venv-tlob/Scripts/python.exe -m jinyiquant.baselines --plan configs/later_date_baselines.json --output runs/later-date-baselines/attempt-001
```

正式计划中的 `pilot_run` 指向已完成且完整性校验通过的短测档案；更换机器或训练设置时，应重新短测并创建新计划。
当前计划保留现有 TLOB 主干、原始官方收益标签、Adam/MSE、两轮训练和09:30/14:30时点；
每段训练两个交易日，研究验证为2023-07-18至07-21、07-25至07-28，种子11/29/47。
8月3日最终区间及预取边界均自动检查，最终区间不加载。7月已做过学习诊断，属于研究数据。
Mean/Ridge/TLOB分别训练，使用相同训练键、评分键、标签、覆盖阈值和单次时间/内存上限；
Mean/Ridge为闭式解，无需人为消耗与神经网络相同的轮次，重复种子也不增加独立日期证据。
最多两个独立单线程运行并发，总时间上限7200秒、总内存10240MiB，单次上限900秒/6144MiB。
额外复跑预先指定时段的首个种子TLOB，核对保存预测及指标容差，不选模、不替换最佳权重。

自动输出 `summary.json/csv`、`metrics.json`、`daily.csv`、`by_timestamp.csv`、
`coverage_by_timestamp.csv`、`paired_comparison.json` 和 `baseline_overview.png`。
每个子运行含训练/验证定权误差曲线 `learning_curve.json/csv/png`、原始预测、评分预测、历史质量和资源日志。
闭式模型曲线仅表示训练均值参照与一次求解结果，不能解释为逐轮优化。
常数预测的IC、RankIC和高低分组/换手在此入口均为未定义；CSV留空、JSON用null。
信号变化包括共同股票上的分数绝对变化、排序持续性及组合净权重换手；首时点未定义。
缺失报告分别列出答案缺失、无历史、min_history不足、完整128条不足、含补齐及源字段缺失；
训练历史质量以有答案的训练键为分母，验证历史以独立完整预测股票池为分母，答案覆盖始终以独立股票池为分母。
逐日均值跨种子后按连续交易日块配对抽样；四天一期证据有限，指标都是本地近似口径。

### 模型学习能力诊断

在独立 TLOB 环境执行两个预先冻结的小实验：

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-tlob/Scripts/python.exe -m modified_tlob.diagnostics --plan configs/learning_diagnostics.json --output runs/learning-diagnostics/attempt-001
```

配置冻结32个完全观测的真实训练样本，沿用原始官方收益尺度、主干、Adam和学习率，固定500次更新。
最终误差必须同时低于初始化误差和样本均值预测误差的10%，防止只学会截距就通过；这只是记忆诊断门槛，不是收益改善标准。
保留学习曲线、目标尺度、梯度范数、各层权重变化、输入/样本、保存加载及同种子复跑检查；未通过也如实完成实验并记录结论。

负对照在2023年7月两段研究窗口、三个种子上比较正确答案和同一完整预测时间戳内固定打乱的训练答案。
只有实际参与训练的股票进入置换，逐截面标签分布保持不变；评估用后续日期的原始答案与独立股票池。
配对固定模型初始化、输入顺序、训练预算和评价键，记录训练前预测作为额外参照。
输出逐日配对指标及连续交易日块区间；两个时段均出现跨种子的显著正向打乱标签RankIC时提示排查。
每段仅四个交易日、三个置换，统计证据有限；未触发提示不证明无泄漏，也不证明模型有效。
冻结的8月3日最终区间不读取，不选模或替换权重。总预算2400秒/4096MiB，由独立进程监控。
产物为 `manifest.json/events.jsonl/metrics.json/small_sample.json/negative_control.json/paired_daily.csv`、子运行及可恢复源码，已有目录拒绝覆盖。
完成后可独立复算相关指标、核对预测键，并输出训练误差相对均值预测的比例和两版预测的相关性：

```powershell
& .venv-tlob/Scripts/python.exe scripts/analyze_learning_diagnostics.py runs/learning-diagnostics/attempt-001 --output runs/learning-diagnostics/audit-001
```

该检查只读已保存产物，不重开原始数据；独立档案保留复算代码与检查结果。
运行中断后可向原诊断命令增加 `--resume-from <旧档案目录>`，并指定新的输出目录；
只复用校验完整且已完成的子运行。冻结计划、核心训练/数据/评价源码、样本与置换必须一致，
未完成配对重新训练，每次恢复使用独立预算，原始失败或超时状态保留。
方法参考 [单批次过拟合检查](https://karpathy.github.io/2019/04/25/recipe/)与[分组内置换标签](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)。

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
| `change` | 单一 `factor`、基线值 `baseline`、候选值 `candidate`、非空 `hypothesis` |
| `threshold_basis` | 校准档案路径及摘要、实际意义门槛 `practical_improvement`、基线波动倍数 `noise_multiplier` |

例如只研究历史长度时，登记 `change={"factor":"data.lookback","baseline":30,"candidate":60,"hypothesis":"更长历史改善后续日期的预测"}`。一轮只能声明一个模型或数据字段，不能同时改变模型结构、窗口和字段。支持历史长度、历史预取天数、最少历史、字段集合及缺失处理等数据因素；评价时点、标签定义、股票池、数据版本和来源、时间切分、预算及评价口径仍固定。评价日期、样本键和答案摘要必须相同，不能用删掉难预测样本制造改善。增加字段时允许对应字段元信息变化，原始文件校验仍须一致；新增字段应来自双方共同冻结的数据快照。`control.identity`（两值均为 null）仅供相同方案的工程负对照。

`threshold_basis` 例如 `{"calibration_report":"calibrations/校准目录/calibration.json","calibration_sha256":"校准JSON内容的SHA256","practical_improvement":0.001,"noise_multiplier":1.0}`。摘要按项目的规范 JSON 计算。工具复核已完成校准档案的产物、配置、种子/时段网格，并从逐日结果重算波动；校准必须早于策略登记。`min_improvement` 至少为 `max(practical_improvement, noise_multiplier × 基线跨种子/时段标准差)`，两个依据均须为正数，倍数及实际意义由研究目标预先说明。这里的0.001和1.0只是格式示例，不能直接作为研究阈值。门槛验证失败时不运行 suite；离线对照缺失可验证校准时输出证据不足。suite 的校准路径必须与策略一致，所有路径相对于所在计划或策略文件解析。

主指标或次要指标涉及 RankIC/IC 时，对照必须覆盖全部由标签决定的可评价截面，并同时核对截面数和完整时间戳的 SHA256。标签为常数的截面不纳入相关评价；候选在可评价截面输出常数导致覆盖不足时拒绝，基线覆盖不足或旧档案缺少覆盖证据时结论为 `inconclusive`。描述指标仍保留未定义相关性，不能用忽略这些值后的均值证明改善。

```powershell
uv run jinyi suite <suite.json>
```

Suite 先冻结策略、假设和预算，再为每个种子/时段分别执行基线、基线复验、候选、候选复验；单次失败即停止且保留全部尝试。比较数据、时间切分、预算、样本、答案、源码、评价口径及成本；通过真实复跑产物验证容差。按交易日配对连续日块 bootstrap，同一时段跨种子使用相同抽样，不把同日股票当独立样本。

`decision.json/result` 统一为三类：**保留**（`accept`）：达到预登记改善门槛、置信区间支持，复验和成本检查均通过；**回退**（`reject`）：证据完整但改善未达标，或关键指标/行为/成本检查失败；**证据不足**（`inconclusive`）：日期、种子、复验、校准或区间支持不足。缺少复验且观察改善不足时不能据此下回退结论；已发现明确行为或成本失败则回退。`action` 自动选择保留候选或维持基线，所有历史产物保留，不覆盖最佳模型。多次筛选仍有选择偏差，需新的验证证据。

先利用已有学习审计提出下一假设：

```powershell
uv run jinyi diagnose --learning-audit runs/learning-diagnostics/audit-001 --output runs/evidence-led-diagnosis
```

输出 `diagnosis.json`、完整校验清单和日志；只读取审计产物，不打开原始数据或最终检查区间。小样本失败时优先排查输入、目标尺度和优化；小样本通过但常规训练差于拟合常数时，可以先单独检验学习率。报告为八类诊断保留证据状态，未测得的分日期、时点、历史覆盖、简单模型等效性、稳定性和盘口贡献不能自动认定成立；“增加成交信息”尚无优先级证据。这只是下一轮提案，仍须预登记和公平复验，不能当作改善结论。公开方法参考 [TLOB 的时序/字段注意力消融](https://arxiv.org/html/2502.15757v3#S7.SS6)与 [MLflow 的基线阈值验证](https://mlflow.org/docs/latest/ml/evaluation)。

## 自动验收与证据

冻结真实小切片验收版本为 `tlob-real-engineering-v1`，配置见 `configs/frozen_tlob_real_v1.json`：
12 个盘口字段、128 条历史、min_history=1、历史预取10天、seed=42、CPU两轮训练，
2023-08-01训练、08-02研究验证，仅抽取09:30和14:30两个完整股票池截面；08-03最终区间不打开。
保存/加载及同配置复跑预先声明 `atol=1e-6, rtol=1e-5`，数据校验值和预测键必须完全一致。
训练单次限300秒/4096MiB，两次训练及独立无标签预测合计限900秒；不据指标修改配置或晋升模型。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-tlob/Scripts/python.exe scripts/real_data_acceptance.py --output runs/frozen-tlob-real-v1/acceptance-001
```

输出 `freeze.json` 固定有效配置、代码提交、可恢复源码及校验、环境和验收规则，
`acceptance.json` 汇总结果，`alignment.json`核对独立股票池、原始官方答案和覆盖；失败档案保留，重跑需新目录。
每个训练档案额外保存 `train_history.parquet/evaluation_history.parquet`：逐键记录填充前有效历史条数（任一特征有效）、
完整历史条数（所有特征有效）、含补齐位置的缺失比例、源记录缺失比例、最近有效行情年龄和窗口跨度。
年龄按自然分钟计，包含午休/隔夜；无历史用null。`coverage.json/history_quality`给出汇总，
min_history达标与完整128条历史分别报告，不把有限前填后的数值当成观测。有限值是当前有效口径，零价不自动改写为缺失。
最终检查模式只在训练结束且权重保存后读取final数据，final答案不进入逐轮验证；行为测试检查实际加载顺序。
独立解释器执行只推理入口并禁止读取答案源/字段，比较预测和档案校验。本地CPU小切片通过仅证明工程流程，
不证明模型收益改善、全量预算或官方平台已验收；当前 CUDA 研究运行另见 `runs/training-progression/`。每次运行原始预测、近似指标及源码快照长期保留。

```powershell
uv run ruff check .
New-Item -ItemType Directory -Force scratch | Out-Null
uv run pytest -q --basetemp scratch/tests-001
uv run python scripts/acceptance.py --output runs/engineering-acceptance
```

验收脚本从头生成固定种子数据，执行正常运行、相同配置复验、超时、超内存、缺文件、无效配置脱敏、基线波动测量及多种子/多时段对照。候选故意与基线相同，完成充分证据后必须得出“回退”，用于检查系统不会把零改善误判为成功。脚本不覆盖已有输出；重复运行须换新目录。当前系统临时目录权限有限，因此测试明确使用工作区内的新临时目录。

`audit.sqlite3` 是事务化运行/最终查看索引，`index.jsonl` 为便利索引。每次运行的日志、配置、源码、指标、检查和模型持久保留；无自动删除历史/最佳版本策略。大数据压缩包、派生缓存和 runs 不纳入 Git，保留源数据校验信息；需要长期异地归档时应整体复制验收通过的运行目录及匹配数据。Ridge/Mean 不使用 GPU；modified_tlob 的 CPU/GPU 本地工程检查与真实7月训练记录见模块说明。

具体本地指标口径保存在每次运行的 `metrics.json/conventions`。公开实现核对参考 [SciPy 的常数输入相关性约定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)与 [NumPy 的缺失值均值说明](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html)；对照额外要求固定的标签可评价截面集合。工程记录保留在独立运行档案中。

## 固定三种子等权集成

固定三种子等权集成研究已完成：两窗分别使用2023-11-01至11-14、2024-03-01至03-14，各训练seed11/29/47一次，共6次1500步训练。预测还原原始标签尺度后直接平均，seed11同时作为参考。全天RankIC平均差为+0.007928，超过旧校准门槛0.007070；两窗改善，5日块95%区间为[0.006151, 0.011038]，2日块敏感性分析同向。但平均换手增加0.012729，违反预登记的“平均换手不升”护栏，因此**不采纳集成，保留seed11参考**。毛多空与最差日均改善，不据此放宽换手要求。详细结果、失败恢复及档案索引见[结构化研究记录](modified_tlob/fixed_seed_ensemble_20261017.json)。只使用两个研究窗，确认、锁定及旧单日final未打开；本地近似指标不等于官方成绩。

该轮使用独立`.venv-ensemble-cuda`环境和`modified_tlob/requirements-ensemble-cuda.lock.txt`。训练与推理入口为`python -m scripts.run_seed_ensemble run configs/fixed_seed_ensemble_20261017_plan.json`，会拒绝重复消费已登记训练次数；评分失败时用`recover-analysis <原执行档案> --output <新目录>`只恢复评分。对已完成的分析档案可只读复算，不训练、不读取原始行情/标签文件：

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.audit_seed_ensemble runs/fixed-seed-ensemble-20261017/analysis-recovery/20261017T091219.300500Z_7c40b5b5d26d --output runs/fixed-seed-ensemble-20261017/offline-audit
```

## 2022 全年诊断与云端运行

全年诊断固定当前 seed11、十个交易日、八时点、1500 次更新配方。固定方案使用 2021 年末训练的同一份权重预测 2022；月度方案每月只用此前十个交易日训练并预测随后一个月，一月份共享训练与预测，共 12 次训练。现有 2023/2024 权重不用于回测更早日期；2024 年 7 月、第四季度及旧 final 保留。

`scripts/prepare_annual_data.py --archives-dir <原包目录> --output-dir <新数据目录>` 只展开 2021-12 和 2022 年所需月份，保留来源与产物校验值。Linux 需要 `bsdtar` 或 `unrar`。先运行 `python scripts/cloud_preflight.py --output <新的检查文件.json>`，CUDA 前向与反向检查成功后再启动训练：

```bash
PYTHONPATH=src:. python -m scripts.annual_baseline run <新的运行目录> \
  --data-root <新数据目录> --cache-dir <缓存目录> \
  --max-rss-mb 40960 --job-wall-seconds 3600 --total-wall-seconds 28800
```

`prepare` 子命令可只冻结数据日历和配置，不训练；创建计划前须部署最终源码，续跑会核对源码、数据、权重和预测摘要。结果包含逐日×八时点 RankIC、IC、多空、换手、误差、覆盖率、权重日龄及两种方案配对差；每月 HTML 与月×时点分布可定位失败，**不输出全年平均数**。跨月换手保持连续，收益为未计费的本地标签近似口径。

2026-10-17 云端诊断已完成：12 次训练、242 个交易日，两种方案各 1936 个时点。完整的 51 列、3872 条逐时记录按月保存在 [研究结果索引](modified_tlob/annual_baseline_2022_20261017/index.json) 所列的 12 个 JSON 中，指标与源 CSV 逐单元格一致；空值转为 `null`，保留未定义标志。索引同时记录训练窗口、配置、权重身份、代码和数据校验值。完整 HTML、CSV、权重与运行档案保留在本地及云存储。

本轮按月定位到的主要问题如下，均为描述性中位数或负截面计数，不是显著性结论：

- 固定权重：4 月毛多空负截面为 58/152，10 月为 48/128。
- 每月重训：5 月 RankIC 中位数为 0.007942，负截面 67/152；6 月毛多空负截面为 72/168，对应固定权重为 45/168。
- 失败不限于开盘：每月重训在 9 月 14:30 的毛多空负截面为 14/21；10 月 13:30 的 RankIC 中位数为 -0.022629，RankIC 和毛多空均有 11/16 个负截面。
- 12 月出现不同表现：每月重训与固定权重的 RankIC 中位数分别为 0.052383、0.040920，换手中位数分别为 1.341709、1.482412。单年份、单种子且权重日龄与月份混杂，不能据此作因果判断或自动替换当前模型。

## 10日与40日训练覆盖研究

本轮冻结方案以 [v3配置](configs/training_coverage_20261017_v3_frozen.json) 为准；此前两版草稿方案已删除，其副本仍留在本地 `runs/training-coverage-freeze/` 的两份档案中。唯一研究变量是训练日期数10→40；raw12、八时点、128条输入、MSE、train_std、1500步、batch32、学习率1e-4、无平滑保持固定。研究种子11/29/47全部登记，部署种子固定11。训练矩阵为每个起点2方案×3种子=6次训练（全轮24次），同起点同方案的三份配置除`seed`外完全相同；每次训练完成后冻结权重、只做推理与评价，不按成绩重训或挑种子。此轮不启动模型训练，数值晋升门槛和40日训练资源预算留待基线校准后另行冻结。

训练清单由官方股票池日历生成，沿用原登记截止日：研究A为2023-11-01，研究B为2024-03-01，确认为2024-06-29，锁定检查为2024-10-01。研究A/B已被使用，只能用于初筛、校准和复验；2024年7月确认区、第四季度锁定区和旧final继续保留。锁定方案配置的validation段是国庆休市间隙，只用于配置结构，不作为额外评价区。

`python -m scripts.freeze_training_coverage` 保存8份训练目标清单、逐项剔除计数、官方源校验值及完整运行档案，拒绝覆盖已有方案。官方标签下载没有 `label_end`：有实际结束戳时优先据其清除跨界样本；无结束戳时，利用官方“不预测隔夜收益”约束，以当日15:00为保守上界，核验早于所有午夜训练截止日。原11:30→13:30选择时钟仍标为假设，不能冒充已验证的实际午休结束时刻。

低成本诊断入口（先按模块README配置独立TLOB环境和`PYTHONPATH`）：

```powershell
python -m scripts.diagnose_training_coverage history
python -m scripts.diagnose_training_coverage inference --annual-root runs/annual-2022-cloud-source/original --device cuda
python -m scripts.diagnose_training_coverage inference --annual-root runs/annual-2022-cloud-source/original --device cuda --resume-from <中断的季度推理档案目录>
python -m scripts.publish_training_coverage --history-run <历史质量档案目录> --inference-run <季度推理档案目录>
```

季度诊断只使用与全年索引校验一致的1/4/7/10月原权重，复用6组原预测，补6组推理；不接受本机另一次训练产生的同名权重。`--resume-from` 只接受同一冻结索引、同一`jobs.json`的中断档案，并逐月校验日历、八时点、无重复和非有限得分后复用已保存预测，避免重复消耗补充分组合预算；复用月份及其校验值写入`checks.json`。历史诊断复用`history_diagnostics`；旧全年输出缺少逐时完整长度及年龄明细，因此仅重放无标签的历史窗口构建，不运行模型。结果按[逐月研究索引](modified_tlob/training_coverage_20261017/index.json)组织，包含逐时排序、多空、换手、配对差、128条完整性、缺失率和最近有效行情年龄，不输出全年平均数。单年单种子且训练起点、行情和权重日龄混杂，不作因果推断，不据此晋升40日方案。

## 仓库发布边界

仓库只包含代码、测试、配置模板、依赖锁、说明及不含证券级样本的结构化研究报告，不上传行情、股票池、答案、证券映射CSV、缓存、模型或运行档案。`.gitignore` 与 `scripts/check_repository_payload.py` 共同检查；提交前执行 `uv run python scripts/check_repository_payload.py --mode staged`，`--mode tracked` 检查完整仓库文件清单。GitHub Actions 在 Windows/Linux 检查依赖、静态规则、行为测试和仓库文件清单。
