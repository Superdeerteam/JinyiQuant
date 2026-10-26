# JinyiQuant 研究工程入口

可选 [modified_tlob 回归模型](modified_tlob/README.md) 使用上游双轴主干、官方收益标签与 MSE，复用统一运行档案。该后端使用独立 Python 3.11 环境，CPU 基础锁为 PyTorch 2.5.0，当前 RTX 5060 研究环境为 2.7.1+cu128；原 Python 3.13 Ridge/Mean 工作流保留。安装、CPU/GPU 验收状态和平台提交包清单见模块说明。

同一后端另有 MLPLOB 主干（`model.kind="mlplob"`）：输入、CSZ 标签变换、目标缩放、MSE、检查点与评价与 TLOB 完全共用，只把注意力主干换成上游双轴 MLP 与单输出收益回归头，不使用位置编码或 `num_heads`。本地 CPU 冒烟用 `configs/mlplob_cpu.json`，V3 同配方的 60 日三种子用 `configs/mlplob_v3_60d_csz_19800.json` 与 `_seed29/_seed47` 变体，运行入口同上。

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

本轮冻结方案以 [v3配置](configs/training_coverage_20261017_v3_frozen.json) 为准；此前两版草稿方案已删除，其副本仍留在本地 `runs/training-coverage-freeze/` 的两份档案中。唯一研究变量是训练日期数10→40；raw12、八时点、128条输入、MSE、train_std、1500步、batch32、学习率1e-4、无平滑保持固定。研究种子11/29/47全部登记，部署种子固定11。训练矩阵为每个起点2方案×3种子=6次训练（全轮24次），同起点同方案的三份配置除`seed`外完全相同；每次训练完成后冻结权重、只做推理与评价，不按成绩重训或挑种子。此轮不启动模型训练，数值晋升门槛和40日训练资源预算留待基线校准后另行冻结（两项启动条件已在下一小节补齐）。

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

### 40日启动条件与研究A/B对照

两个启动条件已用实测补齐，[v4冻结配置](configs/training_coverage_20261017_v4_frozen.json) 与[执行计划](configs/training_coverage_20261018_v4_plan.json) 保留原训练日期、评价窗口和配方，只替换执行预算与决策门槛：

- 预算实测（4次真实预处理路径试点 + 100步优化探针）：10日池预处理31/27秒、峰值2,835/2,929MiB；40日池预处理95/93秒、峰值6,201/7,479MiB。原6,144MiB提案上限被实测证明不足，冻结为9,472MiB与每次训练900秒；样本上限按冻结清单精确设为319,632/319,720，原100,000会直接拒绝启动。六次候选训练实测304–320秒、6,907–7,587MiB，均在冻结预算内。
- 门槛只用六个10日基线校准：窗口级RankIC跨种子标准差0.011647为最小改善量，退化容忍为多空1.0387e-5、换手0.037067、最差日1.3897e-4。冻结时没有任何40日结果。
- 身份一致：六份基线的训练key/标签摘要与冻结清单逐一相同，配方差异仅`experiment_id`/`hypothesis`/`seed`，九个训练源码文件与归档快照逐字节一致，归档逐日指标可由其自身预测完全复算。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_training_coverage_arms budget configs/training_coverage_20261017_v3_frozen.json --prior configs/training_coverage_20261017_v3_frozen.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_training_coverage_arms prepare configs/training_coverage_20261017_v4_frozen.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_training_coverage_arms run configs/training_coverage_20261018_v4_plan.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_training_coverage_arms record configs/training_coverage_20261018_v4_plan.json --execution <运行目录> --output <新记录.json>
```

**结论：回退，保留10日参考。** 池化的配对逐日全天RankIC差（40日减10日，种子均值）+0.012727，形式上越过了0.011647的最小改善量，5日和2日连续日块95%区间分别为[0.003353, 0.016863]和[0.001094, 0.020646]；换手均值下降0.090472，多空均值未退化。但两个研究窗方向相反：研究A为+0.035772（三个种子分别为+0.0409/+0.0152/+0.0512），研究B为−0.010318（−0.0009/−0.0095/−0.0205），因此预登记的`both_windows_improve`与`worst_day_guardrail`（研究B最差日多空差−0.000261，超过1.3897e-4容忍）均未通过。池化改善完全由单一窗口驱动，属于复现失败，不是改善。

**解释边界：** 1500步×32只提供48,000次样本暴露，约为10日池的0.60轮、40日池的0.15轮。本轮只回答"相同计算预算下扩大日期覆盖能否改善泛化"，答案是未得到可复现的稳定改善；这不等于"更多历史数据无效"——40日池在此步数下远未遍历，两个窗口的行情状态也不同，本轮未按规定追加步数或种子。完整成本、逐种子结果、门槛与档案见[结构化研究记录](modified_tlob/training_coverage_arms_20261018.json)。

### 40日方案加步复验（1500→6000步）

上一轮40日方案在1500步（0.15轮）下失败，本轮只检验"这是欠拟合而非旧日期负迁移"这一条假设，[冻结计划](configs/coverage_step_scaling_20261018_plan.json) 与[脚本](scripts/run_coverage_step_scaling.py) 在首个6000步训练前固定：唯一变量是40日臂`max_steps`1500→6000（192,000次样本暴露，0.60轮，与10日臂已获得的份额相同）；归档40日/1500步作步数对照，保留的10日/1500步作参照；网络、学习率、批量、预处理、`target_std`与种子不变。决策门槛沿用上一轮归档值（最小改善量0.011647、两窗改善、多空/换手护栏、最差日容忍1.3897e-4、5/2日连续日块区间），初筛只花2次训练：seed 11两窗都不劣于参照且最差日护栏成立才补29/47。

权重按相同种子从零重训。六次训练在step 1500的进度损失与归档1500步训练**逐位相同**（如research_a s11为1.1277108979419238e-05），因此这确实是同一轨迹的继续，而不是加载旧权重再补4500步。首次尝试在预处理阶段被会话中断、未产生模型，作为浪费尝试记录在案，不复用。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_step_scaling prepare
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_step_scaling run configs/coverage_step_scaling_20261018_plan.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_step_scaling continue configs/coverage_step_scaling_20261018_plan.json --screen <初筛通过档案>
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_step_scaling record configs/coverage_step_scaling_20261018_plan.json --execution <运行目录> --output modified_tlob/coverage_step_scaling_20261018.json
```

**结论：回退，保留10日参考。** 初筛在seed 11通过（研究A +0.039024、研究B +0.007656，四项初筛条件全过）。补足三种子后，配对逐日RankIC差（40日/6000步减10日/1500步，种子均值）+0.020429，越过0.011647门槛；研究A +0.040495、研究B +0.000362，`both_windows_improve`、多空、换手与两种日块区间全部通过。但`worst_day_guardrail`仍失败：研究B最差日多空差−0.000277，超过1.3897e-4容忍（上一轮为−0.000261），集中在2024-03-04与03-11两天，不是系统性尾部；B的逐种子RankIC差为+0.0077/−0.0071/+0.0005，改善未跨种子复现。按预登记本轮到此结束，未再搜索其他步数。

**解释边界：** 步数对照（同池同种子，6000减1500）为+0.007701（A +0.004723、B +0.010680），5日块区间[0.005161, 0.015322]不含零、2日块含零，B在10天中有7天改善——方向支持"0.15轮欠拟合"，但区间在两种块长下结论不一致，证据只到方向性。即使当初通过，也只能说明"额外计算让40日方案更有用"，不能说明长历史优于同计算近期数据；那需要10日/6000步对照。逐种子结果、门槛与档案见[结构化研究记录](modified_tlob/coverage_step_scaling_20261018.json)。

### 40日方案日期衰减损失权重（半衰期10日）

上一轮把40日方案加到6000步后仍只差`worst_day_guardrail`，本轮只检验“旧日期稀释近期状态”这一条假设。[冻结计划](configs/coverage_date_weighting_20261018_plan.json)与[脚本](scripts/run_coverage_date_weighting.py)在首个加权训练前固定：40日池、6000步、batch 32、种子11/29/47、原批次顺序、全40日`target_std`均不变，唯一变量是每个样本的MSE权重由1改为`2**(-交易日年龄/10)`，最新训练日年龄0，权重在所选训练池归一化为均值1；最近10日权重占比53.34%（等权为25%）。只改损失权重，不换采样器。等权40日/6000步归档作为机制对照；晋级仍对照10日/1500步，门槛、护栏和两种日块区间沿用归档值。种子11只用于记录初筛，三种子必须全部完成，不能据种子11认定改善。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_date_weighting prepare
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_date_weighting run configs/coverage_date_weighting_20261018_plan.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_coverage_date_weighting record configs/coverage_date_weighting_20261018_plan.json --execution <运行目录> --output modified_tlob/coverage_date_weighting_20261018.json
```

六次训练均完成6000步，权重均值为1、最近10日占比0.5334/0.5335，`target_scale`与全40日训练标准差一致。**结论：回退，保留10日参考。** 种子11初筛通过（研究A +0.038753、研究B +0.008696），但三种子配对逐日RankIC差（加权40日减10日/1500，种子均值）+0.010851，低于未改的0.011647门槛；研究A +0.027178、研究B −0.005476，`both_windows_improve`不成立；研究B最差日多空差从等权的−0.000277收窄到−0.000239，但仍超过1.3897e-4容忍，`minimum_mean_gain`与`worst_day_guardrail`同时失败。直接对照等权40日/6000步的加权效应为−0.009578（研究A −0.013318、研究B −0.005838）；6个种子×窗口单元中只有B seed11微正（+0.001040），A seed11和两个窗口的seed29/47均退化，说明本轮10日半衰期没有改善等权40日方案。按预注册未搜索其他半衰期。完整逐种子结果、权重统计、门槛与档案见[结构化研究记录](modified_tlob/coverage_date_weighting_20261018.json)。

**解释边界：** 结果只否定“固定10日半衰期、mean-1、6000步”这一组合，不否定其他衰减强度或正则化方式；多数种子-窗口单元退化说明不能靠seed11的初筛通过认定改善，但两个研究窗仍是已使用数据，不能当作独立确认。最终检查区间未打开，平台时限与内存仍未验收。

### 10日标签按完整时间戳去均值（20261018）

上一轮及更早复验反复出现“seed11通过、完整复验失败”。本轮只检验一条最小假设：训练标签改为`y* = 官方收益 − 同一完整时间戳的训练股票池平均收益`，按完整截面一次计算，不按随机小批量；`target_std`仍取原始训练标签标准差，网络、输入、1500步、batch顺序、MSE、三种子×两窗口均不变。推理直接输出相对收益分数，不读取标签、不补回真实未来截面均值；全部评价继续使用原始官方标签，原始MSE与中心化损失分开报告。该“只去均值”做法有[Qlib逐时间戳截面处理](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py)的公开先例，但只是本项目的最小假设，不是已验证有效的配方。

[冻结计划](configs/label_centering_20261018_plan.json)在首个训练前固定六次训练、原改善门槛0.011647与最差日容忍1.3897e-4；seed11只记录进度，六次训练仍全部完成。首次执行在分析阶段失败：诊断脚本用float32做时间戳去均值，两个近似并列标签换位，改变了2024-03-12的RankIC；没有重训，诊断改为float64后从原六份拟合恢复，失败档案保留。

**结论：回退，保留10日参考。** 三种子配对逐日全天RankIC差（去均值减原始10日/1500，种子均值）+0.043121，高于未改门槛；研究A +0.060710、研究B +0.025531，`both_windows_improve`成立。多空均值+0.000316959、换手均值−0.193395，5日/2日连续日块RankIC区间[0.029187, 0.052070]和[0.027434, 0.054341]均不含零。但研究B最差日多空差−0.000250604（2024-03-07）超过1.3897e-4容忍，`worst_day_guardrail`失败，因此按预登记回退。误差分开报告（研究A/B，种子11）：原始标签MSE 1.0326e-5/1.5517e-5，参考1.0729e-5/1.9442e-5；中心化标签MSE 8.785e-6/1.130e-5，参考9.522e-6/1.550e-5；训练中心化损失1.240e-5/1.483e-5。逐种子结果、档案与解释边界见[结构化研究记录](modified_tlob/label_centering_20261018.json)和[完成档案](runs/label-centering-20261018/20261018T092920.229347Z_208fdbce4f22)。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_centering prepare
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_centering run configs/label_centering_20261018_plan.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_centering recover configs/label_centering_20261018_plan.json --execution <首次执行目录>
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_centering record configs/label_centering_20261018_plan.json --execution <完成档案目录> --output modified_tlob/label_centering_20261018.json
```

**解释边界：** 本轮只回答“在当前10日、1500步、原始缩放和两个研究窗口下，完整时间戳去均值是否同时通过原门槛与最差日护栏”，答案是否定的；不否定其他截面标签变换、其他缩放口径或最差日容忍以外的评价方式。两个研究窗仍是已使用数据，确认、锁定和旧final未打开，平台时限与内存仍未验收。

### 10日标签截面z-score并匹配中心化方差（20261018）

上一轮的完整时间戳去均值把平均RankIC抬高约0.043，却因研究B最差日护栏回退。本轮只检验一条更窄的假设：同一完整时间戳内先做截面z-score，再用一个训练池常数`c=std(z)/std(q)`把目标方差匹配到中心化标签的全池方差，使各截面的目标幅度可比；`target_std`仍是原始训练标签标准差，网络、输入、1500步、batch顺序、MSE、三种子×两窗口均不变。按冻结公式，`r=float64(float32(y))`，`z=r−μ_t`，`q=z/σ_t`（`σ_t=0`时保留样本并令`q=0`），`c=std(z)/std(q)`，`y*=float32(c q)`，所有标准差`ddof=0`且先于随机批次和历史掩码一次计算。推理仍为`network(x)*target_scale`，不乘回`c`，不补未来`μ_t、σ_t`；缺历史回退值沿用中心化对照的截距。该截面z-score有[Qlib CSZScoreNorm](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py#L305)的公开先例，但“全池方差匹配”是本项目的对照设计，不能称为直接复刻Qlib配置。

[冻结计划](configs/label_cszscore_20261018_plan.json)、[脚本](scripts/run_label_cszscore.py)与[行为测试](tests/test_label_cszscore.py)在首个训练前固定六次训练、原改善门槛0.011647、最差日容忍1.3897e-4、6次有效训练/最多12次启动和9000秒整轮上限；原始10日参考与中心化对照的六份权重、预测、训练键、原始标签和配置在准备阶段核对后复用，不重训。seed11只记录进度，六次全部完成后才判定。六个任务均一次完成，1500步、48,000次样本暴露，实际峰值RSS约2,930 MiB，执行阶段（训练、推理与分析）合计1,841秒，无重试。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_cszscore prepare
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_cszscore run configs/label_cszscore_20261018_plan.json
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_cszscore recover configs/label_cszscore_20261018_plan.json --execution <已有六份拟合目录>
& .venv-ensemble-cuda/Scripts/python.exe -m scripts.run_label_cszscore record configs/label_cszscore_20261018_plan.json --execution <完成档案目录> --output modified_tlob/label_cszscore_20261018.json
```

**结论：回退，保留10日参考。** 三种子配对逐日全天RankIC差（CSZ减原始10日/1500，种子均值）+0.046347，高于未改门槛；研究A +0.064981、研究B +0.027713，`both_windows_improve`成立，逐种子也全部为正（A +0.0545/+0.0576/+0.0828，B +0.0187/+0.0516/+0.0129）。5日/2日连续日块RankIC区间[0.032514, 0.055247]和[0.030225, 0.058482]均不含零，多空均值+0.000340、换手均值−0.231877，点条件与两项区间条件均通过。但研究B最差日多空差−0.000275413（2024-03-07）超过1.3897e-4容忍，`worst_day_guardrail`失败，因此按预登记回退；研究A最差日差为+0.000412441。按本轮规则，不因3月7日是否修复单独决定成功，也不得针对该日追加规则。

**相对中心化对照的另行报告（不新增门槛，不参与判定）：** CSZ减中心化，种子均值RankIC +0.003226（研究A +0.004271、研究B +0.002182），5日/2日区间[0.002600, 0.004496]和[0.001032, 0.005950]；多空均值+2.320e-5，换手均值−0.038482。但逐种子方向不一致：研究A为+0.006438/+0.004627/+0.001747，研究B为+0.016413/−0.009816/−0.000052，且2日块多空下界约−2.59e-7。因此“相对原始标签通过点条件”不等于“新增标准化优于中心化”；后者只有小幅度、非跨种子一致的方向性证据。

误差分开报告（研究A/B，种子11）：原始标签MSE 1.033639e-5/1.545138e-5，中心化标签MSE 8.780151e-6/1.122348e-5，实际优化的变换目标损失1.259908e-5/1.479845e-5；三者度量不同目标，不能互相替代。逐种子结果、两组配对差、两种日块区间、长短侧、逐日指标、成本与决策见[结构化研究记录](modified_tlob/label_cszscore_20261018.json)和[完成档案](runs/label-cszscore-20261018/20261018T104425.466761Z_1b8039d5f4e1)。

**解释边界：** 本轮只回答“在当前10日、1500步、原始缩放和两个研究窗口下，该方差匹配z-score目标是否同时通过原门槛与最差日护栏”，答案是否定的；不否定其他标准化强度、其他目标尺度、其他损失或最差日容忍以外的评价方式。确认、锁定和旧final未打开，平台时限与内存仍未验收；不覆盖现有参考权重。

### 三套标签方案的上日暴露等权风险残差重放（20261018）

官方推理 demo 末尾调用 `M.jyc_eval._latest(factor_data=factor_data, show=True)`，并注明“分数经风格剔除后等价于每日单因子，show=True 画绩效图 (IC / 分组 / 压力期)”，说明官方接口确实存在；但它来自平台专有的 `bigmodule` 包（四个本地环境均无法导入），仓库、演示脚本与数据平台文档都没有实现或参数，离线环境也无法查看其去极值参数、标准化、暴露日期、回归权重与缺失处理，比赛说明只写了“去极值 与 标准化。风格剔除：将因子对 BARRA 风险因子回归，取残差作为新因子”。因此本轮先冻结一套**本地近似**口径，再用它统一重放原始标签、中心化、CSZ 三套方案的 18 份冻结预测（3 方案 × 3 种子 × 2 窗，共 20 个研究日、每日八个时点），不训练、不选模、不设晋级门槛。[冻结计划](configs/risk_neutralization_20261018_plan.json)、[脚本](scripts/diagnose_risk_neutralization.py)、[行为测试](tests/test_risk_neutralization.py)、[结构化记录](modified_tlob/risk_neutralization_20261018.json)与[完成档案](runs/risk-neutralization-20261018/20261018T145748.199454Z_ea4a79ac311e)。首轮重放后才发现上述 demo 证据，只修正了计划中的证据文字并重跑，冻结口径与全部数值不变。

冻结口径：暴露只用**严格早于信号日**的最近快照（同一天八个时点共用上一交易日），因为 `cpt_jyc_2026_exposure` 的 `date` 是午夜且 `ret` 就是当日收益（2023 年 98.95% 的样本满足 `ret(t) = float_market_cap(t)/float_market_cap(t-1)-1`，中位绝对差为 0），当日行是收盘后才完整的快照；分数先按 `median ± 3×1.4826×MAD` 去极值并做截面 z-score，再对 10 个风格因子和全部 32 个行业哑变量做**等权 OLS**（无截距，哑变量已张成截距），取残差；`ret`、`weights`、`float_market_cap` 和行业标签不进入回归，缺上日暴露的股票在每个方案中同样剔除、不前向填充。`weights` 在 999 只评估股票上每日合计约 0.209（应在更大全集上归一），进一步说明不能拿它当回归权重。三套方案、三个种子与原始/残差两个版本使用完全相同的样本与设计矩阵：A 为 79,912 行、B 为 79,904 行，每窗 160 个截面；每窗每个时点的设计矩阵 SHA256 在 18 次拟合间完全一致。对未过滤的原始预测重算，18 份档案的 `rank_ic_mean`、`long_short_return_mean`、`turnover_mean` 与历史记录逐位一致；同配置复跑两次的全部结果文件（`metrics.json`、`daily_*.csv`、`sections_*.parquet`、`section_exposures_*.parquet`）字节相同。

| 窗口 | 方案 | RankIC 原始→残差 | ICIR 原始→残差 | 多空均值 原始→残差 | 多空夏普 原始→残差 | 换手 原始→残差 | 最差日 原始→残差 |
|---|---|---|---|---|---|---|---|
| A | 原始标签 | 0.00830→0.00543 | 0.36→0.59 | 4.7e-5→7.1e-5 | 4.2→15.9 | 1.6453→1.6413 | -2.96e-4→-9.4e-5 |
| A | 中心化 | 0.06900→0.05865 | 2.70→3.62 | 5.11e-4→4.16e-4 | 33.1→49.4 | 1.3312→1.3279 | 3.1e-5→2.23e-4 |
| A | CSZ | 0.07328→0.06435 | 3.00→3.86 | 5.14e-4→4.53e-4 | 38.7→47.8 | 1.2234→1.2158 | 1.20e-4→2.22e-4 |
| B | 原始标签 | -0.00052→0.00473 | -0.06→0.40 | 5.5e-5→8.6e-5 | 3.6→11.7 | 1.5792→1.5755 | -2.26e-4→-9.0e-5 |
| B | 中心化 | 0.02501→0.02145 | 0.56→1.27 | 2.25e-4→1.97e-4 | 10.0→19.5 | 1.5065→1.5091 | -3.56e-4→-1.8e-5 |
| B | CSZ | 0.02719→0.02465 | 0.83→1.51 | 2.69e-4→2.29e-4 | 14.9→24.3 | 1.5373→1.5438 | -3.70e-4→-6.5e-5 |

中心化与 CSZ 的 RankIC 在残差化后保留 85%–91%，说明两个变换的主要增量并非风格/行业暴露；残差化后 CSZ 减原始标签仍为 A +0.05891（5 日块区间 [0.04806,0.06976]、2 日块 [0.04785,0.06899]）和 B +0.01992（[0.01445,0.02031]、[0.01147,0.02846]），CSZ 减中心化为 A +0.00569、B +0.00319。中心化与 CSZ 的多空价差随风格剔除收窄 12%–19%，但组合波动下降更多，多空夏普反而升高，最差日明显改善，换手基本持平（A 略降，B 增加不超过 0.007）。风格+行业解释的分数截面方差为 A 15.4%/20.5%/22.6%、B 15.2%/16.9%/17.3%（原始标签/中心化/CSZ）。多空净暴露方面，A 的 CSZ 原始组合 BETA 为 -0.1502、SIZE +0.0223、LIQUIDTY -0.0413、行业均值绝对值最大 0.0258，残差化后分别变为 +0.0052、-0.0016、+0.0002、0.0029；原始标签组合的 BETA -0.0979、LIQUIDTY -0.1045 也收敛到 -0.0059、-0.0018。B 的原始暴露整体更小，残差化后各风格净暴露均低于 0.007。

上一轮触发回退的 2024-03-07 得到单独解释：该日 CSZ 减原始标签的多空差在原始分数下为 -2.78e-4（旧容忍 1.3897e-4），残差化后为 -7.05e-5；中心化对照由 -2.53e-4 收窄到 -3.5e-5。该日的旧门槛失败主要由风格暴露贡献，但**本轮不启用旧门槛、不据此宣布通过**，也不改变上一轮的“保留 10 日原始标签参考”结论。

逐日压力按上一交易日市场波动与采样成交量相对前 20 个交易日中位数分组：研究 A 高波动 3 天的 RankIC 低于低波动 7 天（CSZ 残差 0.0525 对 0.0694），研究 B 的波动分组基本持平、高成交量日更高（CSZ 残差 0.0323 对 0.0214）；中心化与 CSZ 的残差分数在两个窗口的全部 8 个压力分组中 RankIC 均为正，而原始标签组在 B 的高波动/高成交量分组接近零或为负（原始 -0.0035/-0.0025，残差 0.0020/-0.0005）。逐日、逐时点与分组明细见运行档案的 `daily_*.csv`、`sections_*.parquet` 和 `metrics.json`。

工程复现：

```powershell
$env:PYTHONPATH = 'src;.'
& .venv/Scripts/python.exe scripts/diagnose_risk_neutralization.py run configs/risk_neutralization_20261018_plan.json
& .venv/Scripts/python.exe scripts/diagnose_risk_neutralization.py record <运行目录> --output modified_tlob/risk_neutralization_20261018.json
```

**解释边界：** 官方实现的去极值强度、标准化、暴露时点、回归权重与缺失规则仍未知，本轮结论只对“上日暴露 + 等权 OLS + 上述预处理”这一冻结组合成立；若官方用当日收盘暴露或市值加权回归，数值会变。研究 A/B 已被用于方案选择，跨方案对比是描述性证据，不是独立确认；多空为未计费毛收益，两个十交易日窗口和三个种子只支持日度块重采样区间。确认区、第四季度锁定区与旧 final 未打开，平台复现与时限仍未验收。

## 模型版本 V0/V1 与标签截面标准化启用（20261023）

当前项目冻结为 **V0**：10 个训练日（2023-10-18 至 11-01）、八时点、原始 12 列盘口、lookback 128、hidden 40/4 层、batch 32、学习率 1e-4、1500 步、seed 11、`target_scaling=train_std`、**原始官方收益标签**。[V0 配置](configs/model_v0.json)。

**版本身份更正（20261023）：** 这里的 V0 是 2023-10-18 至 10-31 的本地重训（权重 `1c557b41…`），**不是**首次公榜 0.18503 使用的七月提交权重。提交包 `dist/v0-submission` 的权重 sha256 为 `eb4d6ffb…`，其 `target_scale`/训练均值与 2023-07-11 至 07-24 的标签矩精确一致，provenance 指向 `july_25_28_s11_d10_t8` 的 1500 步 seed-11 run。因此“V1 高于 V0”是相对十月重训的提升，不能读作相对提交版本的提升；三窗固定权重重放显示 V1 相对提交 V0 在 2023-11、2024-01、2024-03 的 RankIC 分别低 0.0121、0.0145、0.0199，V2-60d 则高 0.0075、0.0008、0.0040（区间含 0）。提交包原有三个 payload 文件与 provenance 不一致（缺 requirements.txt，另两个仅 JSON 序列化不同），已按记录内容恢复并验证 6/6。证据见[身份表](modified_tlob/model_version_identity_20261023.json)、[重放记录](modified_tlob/model_version_replay_20261023.json)、[逐日对照](modified_tlob/model_version_daily_comparison_20261023.json)与[平台包重放](modified_tlob/platform_replay_20261023.json)。

**V1 只在 V0 上启用已有的标签截面标准化**：`model.target_centering=timestamp_zscore_train_rms_matched`，即每个完整训练时间戳内 `q=(r−μ_t)/σ_t`（`σ_t=0` 时 `q=0`，ddof=0），再用训练池常数 `c=std(z)/std(q)` 匹配中心化标签方差；`target_std` 仍是原始训练标签标准差。网络、输入、训练日期、步数、种子与预算逐字段一致。[V1 配置](configs/model_v1.json)。

两版一致性由[行为测试](tests/test_model_versions.py)自动检查（只允许 `model.target_centering` 与实验身份字段不同），并用合成 fixture 证明中心化目标能走统一入口。统一 runner 对非 `none` 的中心化目标跳过原始标签验证诊断（[runner.py](src/jinyiquant/runner.py)），因为适配器不接受混合尺度的验证目标；推理仍为 `network(x)*target_scale`，不乘回 `c`、不读取未来标签或未来截面。

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-ensemble-cuda/Scripts/python.exe -m jinyiquant run configs/model_v0.json
& .venv-ensemble-cuda/Scripts/python.exe -m jinyiquant run configs/model_v1.json
& .venv/Scripts/python.exe scripts/record_model_versions.py --v0-run <V0运行目录> --v1-run <V1运行目录> --output modified_tlob/model_versions_20261023.json
```

同一入口、同 seed、同日期、同 48,000 次样本暴露的结果（研究A，10 个验证日、80 个截面，本地近似口径）：

| 版本 | RankIC 均值 | RankIC IR | 多空均值 | 多空夏普 | 换手均值 | 最差日多空 |
|---|---|---|---|---|---|---|
| V0（原始标签） | 0.006586 | 0.269732 | 2.7e-05 | 1.802846 | 1.666915 | -2.89e-04 |
| V1（截面标准化） | 0.061074 | 2.525518 | 4.35e-04 | 28.134921 | 1.308031 | +6.0e-05 |

配对逐日（V1−V0）均值：RankIC +0.054488、多空 +4.08e-04、换手 −0.358884；最差日差为 11-02 的 RankIC −0.012886、多空 −1.37e-05。两版保存的网络权重与归档的原始标签/CSZ 种子11 手臂逐张量完全相同（最大绝对差 0），因此 V1 就是此前已验证过的 CSZ 模型，本轮只是把它变成能由统一入口复现的版本。[结构化版本记录](modified_tlob/model_versions_20261023.json)。

**V1 多种子证据（3 种子 × 2 窗，复用归档）：** 用已冻结的 CSZ 轮六份 fit 作为 V1 的多种子扩展（V1 seed11 研究A 权重与归档 CSZ 臂逐张量相同，V0 同理对应 raw 臂），不重训；逐格 RankIC 已用归档预测与官方 `vwap_return` 独立复算一致。6/6 个种子-窗口单元 RankIC 均高于 V0；种子均值 research_a 0.008284→0.073265（+0.064981），research_b −0.000521→0.027191（+0.027713），20 日配对均值 +0.046347，5/2 日块 95% 区间 [0.032514, 0.055247]/[0.030225, 0.058482]；多空 +0.000340，换手 −0.231877。旧预登记门槛的 `worst_day_guardrail` 仍未通过：research_b 最差日多空 −0.0002754，低于 −1.3897e-4 容忍（2024-03-07）。因此 V1 是平均意义上的领先版本，不是正式门槛通过；确认/锁定/final 未打开。证据见[多种子报告](runs/model-versions/multi-seed-20261023/report.txt)与[结构化记录](modified_tlob/model_versions_multi_seed_20261023.json)。

**风格处理后的表现（复用[三套方案风险残差重放](modified_tlob/risk_neutralization_20261018.json)的三种子窗口均值，不重算）：** 上日暴露、去极值+截面标准化后对 10 个风格因子与 32 个行业哑变量做等权 OLS 取残差。

| 窗口 | 版本 | RankIC | ICIR | 多空夏普 | 换手 | 最差日多空 |
|---|---|---|---|---|---|---|
| A | V0 原始→残差 | 0.0083→0.0054 | 0.36→0.59 | 4.2→15.9 | 1.645→1.641 | -2.96e-04→-9.4e-05 |
| A | V1 原始→残差 | 0.0733→0.0643 | 3.00→3.86 | 38.7→47.8 | 1.223→1.216 | +1.20e-04→+2.22e-04 |
| B | V0 原始→残差 | -0.0005→0.0047 | -0.06→0.40 | 3.6→11.7 | 1.579→1.575 | -2.26e-04→-9.0e-05 |
| B | V1 原始→残差 | 0.0272→0.0246 | 0.83→1.51 | 14.9→24.3 | 1.537→1.544 | -3.70e-04→-6.5e-05 |

残差化后 V1 减 V0 的 RankIC 仍为 A +0.058911、B +0.019919，ICIR +3.274/+1.113，多空夏普 +31.87/+12.67，换手 −0.425/−0.032，两个窗口的最差日 V1 都不差于 V0。V1 的 RankIC 在残差化后保留 87.8%/90.6%，即增量主要不是风格或行业暴露；V0 在 A 只保留 65.5%，B 的原始 RankIC 为负（绝对值比 9.07，比率无方向含义）。风格+行业可解释的分数截面方差为 15.2%–22.6%。

**V1 平台提交包：** 由 V1 训练目录依次执行 `modified_tlob.package`、`--refresh-inference --recover-missing-history` 与[七文件打包](scripts/package_flat_submission.py)生成；`submission_config.json` 中 `weights_sha256` 与 V1 权重一致（`deabe877…`），并显式启用 `recover_then_mean`（开盘缺失历史回补、无历史按训练均值回退）。[离线校验](scripts/check_flat_submission.py)在参考包与扁平包间做逐字节预测比较：普通与 opening-gap 两个场景均通过，预测最大差 0，回补与均值回退已验证，网络被禁用，requirements 与 torch 2.5.0 环境匹配。上传文件为 `submission_config.json`、`provenance.json`、`jinyiquant_flat.py`、`modified_tlob_weights.json`、`requirements.txt`、`inference.ipynb`、`modified_tlob_flat.py`；`dist/` 与压缩包不入库，平台时限与内存仍未验收。


**解释边界：** 新配对只用一个种子和一个窗口，逐日差不是独立确认；风格残差是本地近似口径，不是官方 `M.jyc_eval` 预处理。归档的 CSZ 轮因研究B最差日护栏失败而未被门槛晋升，V1 是按用户方向形成的版本，不记为门槛通过；确认区、第四季度锁定区与旧 final 未打开，平台时限与内存仍未验收。

### V2 60/120 日云端筛选（20261023）

在云 RTX 3090（24GB）/60GB 内存上，用 V1 配方（CSZ 标签、同网络、同 8 时点、lookback 128、batch 32、lr 1e-4、seed 11、validation 2023-11-01..11-14）把训练日期从 10 日扩到 60/120 日；保持 V1 的每训练日曝光 4,800 样本（10 日 1500×32），即 60 日 9000 步、120 日 18000 步，epoch 份额约 0.601。数据按月读取：60 日读 bars 2023-07..11、labels 2023-08..11；120 日读 bars 2023-04..11、labels 2023-05..11。

| 版本 | RankIC | 多空均值 | 换手 | 最差日多空 | RankIC Δ vs V1 | 换手 Δ vs V1 |
|---|---|---|---|---|---|---|
| V1（10 日/1500 步） | 0.061074 | 0.000435 | 1.308031 | +0.000060 | — | — |
| V2 候选 60 日/9000 步 | 0.086028 | 0.000612 | 1.326032 | +0.000274 | +0.024954 | +0.018001 |
| 120 日/18000 步 | 0.088049 | 0.000606 | 1.387195 | +0.000272 | +0.026974 | +0.079164 |

沿用旧校准门槛（RankIC 至少 +0.011647、多空不低于 0.000331、换手不高于 1.345098、最差日不低于 −0.000139）：60 日四项全过；120 日换手 1.387195 超过 1.345098，未过。**V2 = 60 日版本**；120 日 RankIC 虽高 +0.002021，但换手超限，不能按 RankIC 选。单种子、单验证窗，只是筛选证据，不是独立确认；60 日相对 V1 同时有 6 倍更新次数（120 日 12 倍），因此不能把增益全部归因于历史长度。记录见 `runs/model-versions/v2-cloud-20261023/record.json`，两臂权重、预测与指标已下载到同目录；完整云运行目录未随关机保留。

### V1.1 七月训练窗 + CSZ 实验（20261023，仅实验）

用户方向：保留 V2-60d 为基线，用 V1 的 CSZ 配方跑一个“七月 + CSZ、10 日、seed 11、1500 步”的本地实验，命名 **V1.1**，仅用于与 V2 比较，不修改基线。V1.1 训练 2023-07-11 至 07-24、验证 07-25 至 07-28（与提交版 V0 的七月训练窗一致），训练池 79,920 键、48,000 次样本暴露、19/19 行为检查通过，RTX 2060 用时 540 秒。[配置](configs/model_v1_1_july.json)、[回放计划](configs/model_v1_1_replay_20261023.json)、[结构化记录](modified_tlob/model_v1_1_july_20261023.json)、[训练档案](runs/model-versions/v1.1-july/20261024T024915.189199Z_88ac499a27d2)、[回放档案](runs/model-v1.1-replay-20261023/20261024T025927.048023Z_f35e4f24f8a9)。

四个固定权重的同键窗口（本地近似口径，未计费用）结果如下，区间为逐日配对差的 5 日连续块 95% 区间：

| 窗口 | V2 RankIC | V1.1 RankIC | V1.1−V2 RankIC [区间] | V2 多空 | V1.1 多空 | V2 换手 | V1.1 换手 |
|---|---|---|---|---|---|---|---|
| 2023-07-25..28 | 0.088510 | 0.077627 | −0.010882 [−0.015478, −0.006959] | 0.000652 | 0.000521 | 1.3308 | 1.3999 |
| 2023-11 | 0.081449 | 0.067796 | −0.013653 [−0.031528, −0.004509] | 0.000550 | 0.000437 | 1.3245 | 1.3622 |
| 2024-01 | 0.047899 | 0.045035 | −0.002864 [−0.015136, +0.008834] | 0.000396 | 0.000377 | 1.3533 | 1.4119 |
| 2024-03 | 0.075679 | 0.071472 | −0.004207 [−0.014189, +0.003825] | 0.000557 | 0.000480 | 1.3199 | 1.3852 |
| 三个月平均（11/01/03） | 0.068230 | 0.061280 | −0.006950 | 0.000500 | 0.000431 | 1.3328 | 1.3864 |

**结论：保留 V2，V1.1 不晋升。** V1.1 在全部四个窗口的 RankIC、多空均低于 V2，换手均更高；七月和十一月的配对区间不含零，一月和三月含零。相对同配方的十月 V1，七月训练窗把三个月平均 RankIC 提高 +0.0125，但换手增加 +0.0690；相对提交的七月原始标签 V0，V1.1 的三个月平均 RankIC 差 −0.0029（各窗口区间都含零）、换手低 −0.0285，即“七月窗 + CSZ”没有超过原七月原始标签模型，也没有改变 V2 的领先位置。本轮未修改任何 V2 配置、权重、记录、提交包或基线表。

**与 V0/V1 的关系（补充）：** 按版本身份记录，把“新 V0”理解为 2023-10-18 至 10-31 的十月本地重训（raw 标签，权重 `1c557b41…`）；另列七月提交版 V0（raw，权重 `eb4d6ffb…`）作参照。用[补充回放计划](configs/v0_local_oct_common_windows_20261023.json)补齐十月本地 V0 的四窗口，三个月平均（11/01/03，逐日）如下：

| 版本 | 训练窗 / 标签 | RankIC | 多空 | 换手 |
|---|---|---|---|---|
| 新 V0（十月本地重训） | 10-18..31 / raw | 0.013073 | 0.000135 | 1.6673 |
| V1 | 10-18..31 / CSZ | 0.048735 | 0.000362 | 1.3174 |
| V1.1 | 07-11..24 / CSZ | 0.061280 | 0.000431 | 1.3864 |
| 七月提交 V0 | 07-11..24 / raw | 0.064162 | 0.000455 | 1.4149 |
| V2 | 08-01..10-31 / CSZ（60 日） | 0.068230 | 0.000500 | 1.3328 |

V1.1 减各对照的三个月平均配对差：新 V0 为 RankIC **+0.0482**（四个窗口的 5 日块区间均不含零）、多空 +0.000295、换手 **−0.2809**，即 CSZ 相对十月 raw 是主要增量；V1 为 +0.0125、+0.000069、+0.0690（七月和三月显著，十一/一月含零），七月训练窗提高排序但抬高换手；七月提交 V0 为 −0.0029（各窗口区间均含零）、−0.000025、−0.0285，统计上没有差别、未超过提交版。

**工程说明：** 回放脚本的配对列表改为可由计划文件指定，默认仍保持原 20261023 的五个配对不变；新增配对校验测试。三个旧版本的指标与 20261023 回放逐位一致；补充的十月本地 V0 在十一月/一月/三月也与 20261023 回放逐位一致（最大差 0）。V1.1 的七月回放指标与其训练运行相差 9.7e-8（仅 GPU 批大小舍入），说明比较管线可复现。

**解释边界：** V1.1 与 V2 同时改变训练窗长度和更新次数（10 日/1500 步 vs 60 日/9000 步），差值不能归因于单一因素；七月是 V1.1 的验证窗，且 V2 的权重在 8–10 月训练后回看七月，属于描述性证据；所有指标为本地近似口径，不是官方综合分数；确认区、锁定区与 final 未打开，平台时限与内存仍未验收。

### 按月均匀抽取的 132 个完整交易日（20261025，数据读取）

用户方向：在 V2-60d 之外新增按月均匀抽样的训练数据读取，覆盖 2022-01..2023-10 每月 6 个完整交易日、共 132 日，保留三档原始盘口（12 字段）、lookback 128 与 CSZ 标签，按月读取并缓存。设计动机参考 [TLOB](https://arxiv.org/abs/2502.15757)：该文的 dual attention（空间/时间）指出模型须跨不同市场条件泛化，因此用近两年的月度均匀抽样替代单一段连续窗口，让训练池同时覆盖 2022 年下跌与 2023 年前十个月的震荡/回升。本轮只实现数据读取，网络与训练配方不变，**不产生任何指标结论**。

**读取契约（fail-closed，`data.monthly_sampling` + [配置](configs/monthly_132d_csz.json)）：**

- 日历取独立股票池表：某日 8 个采样时点齐全才算“完整交易日”。
- 每月按中心分层索引 `floor((2i+1)·n/(2k))` 抽取 k=6 日；某月完整日不足 6 个直接报错，不静默少取。真实日历核对：22 个月完整日 16..23 个，共选中 132 日。
- 读取仍按月度源文件逐月进行。训练区间内只保留选中日与各自的前一交易日（供 09:30/10:00 的 128 根开盘历史窗使用）；区间尾部的 `history_days` 窗与验证段保持完整，避免首日评估缺历史。
- 标签与股票池同样只保留选中日；训练标签仍走既有 CSZ（`timestamp_zscore_train_rms_matched`），只在训练池内拟合。
- 投影缓存键为“源文件 + 列 + 区间 + 选中日摘要”，逐月落盘；未抽样读取的缓存键与旧版逐字相同，不使历史缓存失效。

**证据（本机 16GB，2022-01..06 六个月子区间冷读实测；[脚本](scripts/verify_monthly_sampling.py)）：** 36 个完整日 / 6 个月，21,470,851 行 bars、367,632 行标签（含 10 日验证尾部）、287,320 个训练样本；每日 8 个时点、每月 6 日全部齐全；09:30 开盘窗 5,994 个样本的历史行数最小 128、满窗比例 1.0、覆盖率 1.0；CSZ 在 288 个训练时间戳上完成（每戳最少 994 行，c=0.0042399，变换后标准差与中心化池一致）。冷读 95.9 秒、复读 69.6 秒且全部 `cache_hit`、`data_sha256` 相同，冷读后 RSS 2.49GB。22 个月的完整选择另在真实日历上核对：132 日、每月 6 日、逐日标签有限比例最低 0.995。

**全量规模与运行边界：** 132 日 × 8 时点 × 8,000 ≈ 1.056M 训练标签；bars 需保留 132 个采样日 + 132 个前一交易日 + 区间尾部，约 70M 行（~8GB 常驻、拼接/排序时约 2 倍峰值），再叠加 `build_batch` 的 ~9GB 已超出本机 16GB，须按 V2 的云端 60GB 执行。配置保持 V2 的 288,000 样本曝光（9000 步 × 32）与全部配方，只改变训练日构成，用于回答“同算力下，月度均匀抽样的状态覆盖是否优于连续 60 日”；该配置尚未训练，没有晋升或指标声明。按每训练日曝光提高到 19,800 步的版本及筛选结论见下节。

### 132 日覆盖 vs 训练量对照（20261025，云端筛选）

用户方向：训练 132 日广历史 TLOB，先按 V2 的每训练日曝光（4,800 样本/日）用 19,800 步；增加相同更新预算的 60 日对照，分清收益来自覆盖面还是训练量；并补跑 V2 60d/9000 的 seed 29、47 以测当前基线波动。云端 RTX A4000（16GB）/60GB 内存，数据为官方下载（逐月 bars sha256 与本地一致），验证窗 2023-11-01..14 不变。

| 臂 | 数据构成 | 种子 | 更新 | epoch 份额 | RankIC | 多空 | 换手 | 最差日 |
|---|---|---|---|---|---|---|---|---|
| V2 60d/9000（归档基线） | 2023-08..10 连续 60 日 | 11 | 9,000 | 0.601 | 0.086028 | 0.000612 | 1.3260 | +0.000274 |
| V2 60d/9000 | 同上 | 29 | 9,000 | 0.601 | 0.083240 | 0.000606 | 1.1913 | +0.000314 |
| V2 60d/9000 | 同上 | 47 | 9,000 | 0.601 | 0.081498 | 0.000562 | 1.0882 | +0.000281 |
| 60d/19800（训练量对照） | 2023-08..10 连续 60 日 | 11 | 19,800 | 1.322 | 0.090481 | 0.000654 | 1.1535 | +0.000423 |
| 132d/19800（覆盖候选） | 2022-01..2023-10，每月 6 个完整日 | 11 | 19,800 | 0.601 | 0.092302* | 0.000633 | 1.3654* | +0.000410 |

\* 该臂训练与推理完成，但在训练后的行为检查阶段被系统 OOM 杀掉（峰值 RSS 50.3GB，60GB 机器同时跑着种子臂），没有 `metrics.json`；数值由该运行保存的 `predictions.parquet` + `targets.parquet` 用同一评估器重算，档案记为失败而非通过。

逐日配对差（5 日连续块 95% 区间，10 个交易日）：

| 对照 | RankIC | 多空 | 换手 |
|---|---|---|---|
| 覆盖效应：132d/19800 − 60d/19800 | +0.0018 [−0.0019, +0.0043]（含零） | −0.000020 [−0.000087, +0.000003] | +0.212 [+0.203, +0.213]（10/10 日变差） |
| 训练量效应：60d/19800 − 60d/9000（seed 11） | +0.0045 [+0.0038, +0.0109] | +0.000041 [+0.000014, +0.000124] | −0.173 [−0.174, −0.163]（10/10 日变好） |
| 总效应：132d/19800 − V2 基线 | +0.0063 [+0.0025, +0.0115] | +0.000021 [+0.000017, +0.000041] | +0.039 [+0.038, +0.045] |

V2 60d/9000 的三种子波动：RankIC 0.083589 ± 0.002285（0.0815–0.0860）、换手 1.2018 ± 0.1193（1.088–1.326）。

**结论：不晋升 132 日版本，保留 V2 60d/9000。** 同预算下 132 日的 RankIC 增益 +0.0018 落在零区间内、且小于基线种子标准差 0.0023，不是覆盖增益的证据；同对照换手 +0.212 全 10 日变差，1.3654 超过 V2 筛选预登记上限 1.3451。真正移动指标的是训练量：60 日窗从 9,000 步加到 19,800 步，RankIC +0.0045（约 2 个种子标准差）、换手 −0.173。132d 相对 V2 基线的 +0.0063 中约 +0.0045 可由「同样的 60 日多训 10,800 步」复现，覆盖部分在噪声内且抬高换手。

**后续三种子复验（20261025）修正：** 该段的「训练量增益」判断建立在单种子 19,800 步之上。补齐 seed 29/47 后，19,800 步的同种子配对差为 +0.0045 / −0.0089 / +0.0071（均值 +0.0009），训练量增益不跨种子复现；份额扫描的完整结论与新基线 V3 见文末「V3 基线」小节。

**边界与下一步：** 两个 19,800 步臂都是单种子；验证窗仅 10 日；本地近似口径，不含费用与冲击；seed 11 基线是 20261023 另一次云端运行的归档预测重算。晋升前需先补 19,800 步的 seed 29/47；若仅训练量效应稳健，应优先「60 日窗 + 更长训练」而不是 132 日组合。配置见 [60d/19800](configs/model_v2_60d_csz_19800.json)、[seed29](configs/model_v2_60d_csz_seed29.json)、[seed47](configs/model_v2_60d_csz_seed47.json)、[132d/19800](configs/monthly_132d_csz_19800.json)；对照脚本 [compare_coverage_volume.py](scripts/compare_coverage_volume.py)，计划与结果 [plan](configs/monthly_coverage_vs_volume_20261025_plan.json) / [comparison](configs/monthly_coverage_vs_volume_20261025.json)，结构化记录 [monthly_coverage_20261025.json](modified_tlob/monthly_coverage_20261025.json)。

### 同一时点成组采样与 MSE+成对排序损失（20261025，实现与预注册）

用户方向：云端容器正在完成 epoch 份额扫描并选出最优模型；随后从 UTC 19:20 起每 20 分钟检查本地是否出现最新最优模型，并以该模型为基准，把训练采样改为“同一时点成组采样”、训练目标改为 MSE+成对排序损失（排序项系数在训练前定好），同时保留同采样器的纯 MSE 对照。参考 [Temporal Relational Ranking](https://arxiv.org/abs/1809.09441) 与[作者代码](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)。

**实现（提交 `d3e459d`）。** `model.sampling=timestamp_grouped` 时，每个优化步只取同一完整时间戳（日期+时点）的截面：按完整时间戳分组，组内打乱后按 `batch_size` 切块，不足一批的组尾丢弃，其余行每轮各用一次；排序项按论文式(8)实现为 `relu(-(s_i-s_j)(y_i-y_j))` 的 `n²` 有序对均值，标签用同批次训练目标（逐时间戳 CSZ 变换不改变同时间戳内的排序）。`ranking_loss_weight` 在训练前固定；`ModelConfig` 校验“有排序项必须成组采样、权重与开关同步、batch≥4”；训练日志新增 `objective_loss/ranking_loss/sampling`，检查点记录损失标签与采样器。[7 项行为测试](tests/test_timestamp_grouped_ranking.py)覆盖分组与批次确定性、hinge 方向及梯度、配置拒绝、两臂日志和检查点回读。

**证据（云端 60 日冒烟）。** 20 步 `smoke-grouped-rank` 完成 79,920 条推理，RankIC 0.021473、换手 1.540946；5 步 `pilot-grouped-60d` 在 60 日训练池记录 `groups=480, samples=479432, samples_used_per_epoch=476160, dropped_tail_per_epoch=3272`（丢弃 0.68% 的组尾），5 个进度步的 ranking_loss 0.031–0.050、MSE 0.49–1.76，GPU 训练与推理均正常。这些只是管线冒烟，不是模型质量证据。

**与作者代码的差异（记录）。** 作者代码的成对项写成 `reduce_mean(relu(+(gap_pred)(gap_gt)))`，与本轮采用的论文式(8)符号相反；本轮按论文公式实现，α 训练前固定为 1.0（作者 CLI 默认值；论文网格 {0.1,1,10}，不按验证集挑选）。

**预注册的后续对照。** 以云端扫描选出的最优模型配方为 base（判定顺序：通过预登记门槛且三种子的臂；否则“≥2 种子、换手≤1.3211”中平均 RankIC 最高者；否则 V2 60d/9000），在相同数据、预算与种子上从零重训两臂：同采样器纯 MSE 对照、同采样器 MSE+1.0×成对 hinge；先 seed 11 成对，再补 29/47，不复用旧权重微调。主要判据：候选−对照的同种子配对逐日 RankIC，种子平均后 5 日块 95% 区间下界>0 且均值增益≥0.0023（V2 基线跨种子标准差 0.002285 的 1 倍）；换手均值≤1.3211，多空均值不低于对照。α 不再调整。

**结果（20261025 完成）。** 云端份额扫描最终选定 V3 = 60d/19800 CSZ 三种子等权集成（[选择记录](modified_tlob/model_v3_20261025.json)）。本实验以 V3 配方为 base，三种子、19,800 步从零重训两臂，另补一条 α=0.1 的单种子探索臂；10 个验证日、本地近似口径：

| 臂（10 日验证） | RankIC | 换手 | 多空 | 预测 std |
|---|---|---|---|---|
| V3 shuffle 三种子集成 | 0.092349 | 1.1274 | 6.49e-4 | 2.2e-4 |
| 同采样器纯 MSE 对照（三种子集成） | 0.090525 | 1.1936 | 6.20e-4 | 2.5e-4 |
| 同采样器 MSE+1.0×hinge（三种子集成） | 0.041647 | 1.3592 | 3.37e-4 | 0（塌缩） |
| 同采样器 MSE+0.1×hinge（seed11） | 0.084731 | 1.2717 | 5.72e-4 | 1.4e-4 |

配对逐日差（5 日连续块 95% 区间）：
- 成组纯 MSE − V3：RankIC **−0.00182** [−0.00345, −0.00102]、换手 +0.0662。同预算下成组采样本身相对 shuffle 可测退化。
- α=1 hinge − 成组纯 MSE：RankIC **−0.04888** [−0.05245, −0.04077]。三个 seed 全部塌缩：验证分数为常数（std=0），训练 hinge 在 1,000 步内降到 1e-3、最终 ~1e-6，而 MSE 轨迹与对照几乎同步，说明排序项把输出压成退化常数而非学到排序。
- α=0.1 seed11 − 同 seed 成组纯 MSE：−0.00113 [−0.00361, +0.00379]（无差别）；相对 V3 seed11 为 **−0.00575** [−0.00973, −0.00374]。α=0.1 不塌缩（训练 hinge 结尾 ~0.013），但未超过同采样器对照。

**决策：不晋升，保留 V3。** 成组采样相对 shuffle 有可测退化；α=1 排序项导致表示塌缩，α=0.1 不塌缩但 seed11 无增益。α=1.0 为训练前预注册值，α=0.1 是在看到 1.0 塌缩后追加的探索臂；seed29/47 于 01:09 UTC 启动，但容器在 02:25 UTC 前不可达，产物未取回，该臂仍为单种子。结构化记录见 [grouped_ranking_20261025.json](modified_tlob/grouped_ranking_20261025.json)，比较脚本 [compare_grouped_arms.py](scripts/compare_grouped_arms.py)，计划 [grouped_ranking_20261025_plan.json](configs/grouped_ranking_20261025_plan.json)。

**边界。** α=0.1 只有 seed11（seed29/47 的云端运行因容器失联未取回产物）；成组两臂未做 2024-01/03 跨窗回放；验证窗仅 10 日、本地近似口径；塌缩原因只定位到“验证输出常数 + 训练 hinge 归零”，未做逐层权重诊断。


### V3 基线：60d/19800 CSZ 三种子集成（20261025，云端扫描收口）

用户方向：把 epoch 份额扫描（0.6 / 0.9 / 1.2 / 1.32 / 1.5 轮，batch-32；外加 batch-64 / 双倍遍历对照）、19,800 臂的三种子与三窗回放跑完，按预登记规则选出最优模型并登记为新基线 V3。

**份额矩阵（2023-11-01..14，79,920 样本，本地近似口径）：**

| 臂 | 种子 | 平均 RankIC ± sd | 换手 | 同种子配对差（逐种子） |
|---|---|---|---|---|
| 60d / 9,000（0.60 轮，V2 基线） | 11/29/47 | 0.083589 ± 0.002285 | 1.2018 ± 0.1193 | — |
| 60d / 13,500（0.90 轮） | 11/29/47 | 0.082971 ± 0.004713 | 1.1677 | +0.0010 / −0.0054 / +0.0026 |
| 60d / 18,000（1.20 轮） | 11/29 | 0.082941 ± 0.002766 | 1.3615 | −0.0011 / −0.0023 |
| 60d / 19,800（1.32 轮） | 11/29/47 | 0.084476 ± 0.008806 | 1.2146 | +0.0045 / −0.0089 / +0.0071 |
| 60d / 22,500（1.50 轮） | 11/29/47 | 0.085596 ± 0.007177 | 1.2116 | +0.0030 / +0.0071 / −0.0042 |
| 60d / 9,900 @ batch64（1.32 轮，半个更新数） | 11 | 0.090189 | 1.2140 | +0.0042 |

**结论一：份额扫描不晋升。** 所有臂的同种子配对差方向不一致；跨种子 sd 随训练加长从 0.0023 涨到 0.0088，说明「加步」买到的是方差而不是质量。预登记门槛（晋升需 3 种子、同种子配对 5 日块 95% 区间不含零且 ≥0.0046、换手 ≤1.3211）无一通过：13,500 与 19,800 三种子均值为负/接近零，18,000 换手 1.362 超护栏，22,500 在两种子时曾以 +0.00509 过门槛，补齐第三种子的 −0.00415 后回落到 +0.00201（仍含反向种子）；batch-64 只有单种子。

**结论二：起作用的是遍历量而不是更新次数（单种子提示）。** 同为 1.322 轮、9,900 步 × batch-64 的 RankIC 0.09019 与 9,900 步 × batch-64 vs 19,800 步 × batch-32 的 0.09048 相差 0.0003，而两者更新数相差一倍。

**结论三：三种子集成是全轮唯一稳定改善，登记为 V3。**

| 集成（等权） | 验证 RankIC | 换手 | 相对 V2 单种子基线（配对区间） |
|---|---|---|---|
| 9,000 × 11/29/47 | 0.089331 | 1.1572 | +0.0033 [+0.0026, +0.0084] |
| **19,800 × 11/29/47（V3）** | **0.092349** | **1.1274** | **+0.0063 [+0.0047, +0.0113]** |

三窗回放（同键、整月、样本外）集成 RankIC：19,800 集成 **0.088735 / 0.053356 / 0.078412**（2023-11 / 2024-01 / 2024-03），对 V2 seed-11 单模型三窗差 **+0.0073 / +0.0055 / +0.0027**，方向全正；9,000 集成为 0.085510 / 0.047100 / 0.077639。

**V3 定义：60d 窗 + CSZ 标签 + batch-32 + 19,800 步（1.32 轮）训练的 seed 11/29/47 三个检查点等权集成。** 注意：19,800 步的*单种子*配方不可单独晋升（种子间 −0.0089~+0.0071），被晋升的是集成；V2（60d/9000）作为上一代基线保留，历史配置、记录与权重不动。

**证据边界：** 13,500/19,800/22,500 为三种子，18,000 两种子，batch-64 仅 seed 11；验证窗 10 日、5 日块区间较宽；三窗均为已用研究窗；集成的推理成本约为单模型 3 倍；本地近似口径，不含费用与冲击；blend 提交包只通过离线 flat 检查（合成 DAI、禁网、含开盘缺历史场景），平台尚未验证。

**产物与复现：** 配方 [model_v3_60d_csz_ensemble.json](configs/model_v3_60d_csz_ensemble.json)，记录 [model_v3_20261025.json](modified_tlob/model_v3_20261025.json)，选型脚本 [select_epoch_share.py](scripts/select_epoch_share.py) + [计划](configs/epoch_share_selection_20261025_plan.json) / [结果](configs/epoch_share_selection_20261025_result.json)，三窗计划 [model_v2_multi_window_final_20261025.json](configs/model_v2_multi_window_final_20261025.json)，提交包 `dist/v3-submission-20261025/`（blend 检查点权重 sha256 前 8 位 `d0cb487a`）。

**MLPLOB 替换主干（20261026）：** 在 V3 数据配方逐项不变（2023-08-01 至 11-01 的 60 个交易日、原始 12 维 L1-L3 盘口、128 根历史、CSZ 标签、train_std、batch-32、19,800 步、seed 11/29/47）的前提下，把 TLOB 注意力主干换成上游双轴 MLP，复用同一训练、缩放、评价与打包路径。

| 三种子等权集成 | 验证 RankIC | 换手 | 多空 |
|---|---|---|---|
| V3（TLOB 主干） | 0.092349 | 1.1274 | 0.000649 |
| MLPLOB 主干 | 0.092936 | 1.1132 | 0.000654 |

**结论：** 配对日度差 +0.00059，5 日块 95% 区间 −0.00323..+0.00398 含零，未过预注册门槛（增益 ≥ 0.0023 且区间不含零），**V3 维持基线**；MLPLOB 作为同口径、可导出、推理更轻的替代主干保留，单种子 RankIC 0.088985/0.084444/0.088076 不作独立结论。

**提交导出接齐：** `research/blend-20261025` 的固定权重集成导出已并入当前主干并适配两种主干：`TLOBAdapter.load` 识别 schema-4 `modified_tlob_blend` 检查点，新增 [build_blend_weights.py](scripts/build_blend_weights.py) 与 [build_blend_package.py](scripts/build_blend_package.py)，`check_flat_submission.py` 支持读取集成成员配置；成员只共享主干类型、预处理、预测尺度与冻结配方键，权重固定、不在推理时拟合或排序变换。MLPLOB 三种子七文件包为 `dist/mlplob-submission-20261026/`（权重 sha256 前 8 位 `d3d8178a`，压缩包 `cc46b0b7`），普通与开盘缺历史两场景离线 flat 校验通过（禁网、notebook 执行、预测最大差 0），平台尚未验收。

**产物与复现：** 配方 [mlplob_v3_60d_csz_19800.json](configs/mlplob_v3_60d_csz_19800.json)（seed29/47 同构变体），对照计划 [mlplob_v3_60d_csz_20261026_plan.json](configs/mlplob_v3_60d_csz_20261026_plan.json)，记录 [mlplob_v3_20261026.json](modified_tlob/mlplob_v3_20261026.json)。


## 仓库发布边界

仓库只包含代码、测试、配置模板、依赖锁、说明及不含证券级样本的结构化研究报告，不上传行情、股票池、答案、证券映射CSV、缓存、模型或运行档案。`.gitignore` 与 `scripts/check_repository_payload.py` 共同检查；提交前执行 `uv run python scripts/check_repository_payload.py --mode staged`，`--mode tracked` 检查完整仓库文件清单。GitHub Actions 在 Windows/Linux 检查依赖、静态规则、行为测试和仓库文件清单。
