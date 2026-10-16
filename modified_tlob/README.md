# modified_tlob：TLOB 主干与官方收益回归

已完成结构迁移和工程闭环，并在2023年7月真实数据开展有预算的训练对照。原论文为十档事件盘口三分类；本项目使用三档/五档分钟盘口、官方 `vwap_return` 和 MSE。原论文 F1 不能当作本比赛的收益预测证据。

已阅读 [原论文 v3 的正文与附录 A/B](https://arxiv.org/pdf/2502.15757v3)，实现以 [官方固定提交 f1c0af4](https://github.com/LeonardoBerti00/TLOB/tree/f1c0af4d81067978914361766db0457a7d8b6a46) 为准。`provenance.json` 保存论文与源码 SHA256、保留结构、必要修复和证据边界；`upstream/*.py.txt` 是行为对照的原始参照，`LICENSE` 原样保留上游 MIT 许可。论文示意图对 BiN 的描述与实现存在粒度差异，这里遵循固定代码的单次输入 BiN。

## 环境与依赖

原项目仍使用 Python 3.13、原 `uv.lock` 和 Ridge/Mean；导入基线路径不加载 PyTorch。TLOB 使用独立 Python 3.11 环境：CPU 基础锁保留 PyTorch **2.5.0**，当前 RTX 5060 研究使用 **2.7.1+cu128**。下面命令从仓库根目录执行，假定已安装 uv（可用 `python -m pip install uv==0.8.22` 安装）。

CPU：

```powershell
uv python install 3.11
uv venv --python 3.11 .venv-tlob
uv pip sync --python .venv-tlob modified_tlob/requirements-cpu.lock.txt modified_tlob/requirements-dev.lock.txt
$env:PYTHONPATH = 'src;.'
$tlobPython = '.venv-tlob/Scripts/python.exe'
& $tlobPython -m pytest -q --basetemp scratch/tlob-tests-001
& $tlobPython scripts/tlob_acceptance.py --output runs/tlob-acceptance-001
```

CPU 锁使用 `torch==2.5.0+cpu`，不需要 CUDA 驱动、Toolkit 或 ONNX。默认运行命令可继续使用 `uv run jinyi ...`；TLOB 在独立环境中用同一 CLI：`& $tlobPython -m jinyiquant ...`。不要在 Python 3.13 环境中安装这份 PyTorch 锁。

RTX 5060 / Blackwell CUDA（在另一个新环境中执行，不能同时安装 CPU/CUDA 锁）：

```powershell
uv venv --python 3.11 .venv-tlob-cuda
uv pip sync --python .venv-tlob-cuda modified_tlob/requirements-blackwell.lock.txt modified_tlob/requirements-dev.lock.txt modified_tlob/requirements-research.lock.txt
```

Blackwell 锁使用 `torch==2.7.1+cu128`，已在本机 Windows / RTX 5060 完成真实7月 CUDA 训练与保存加载检查；旧 `requirements-cuda.lock.txt` 的 2.5.0+cu121 不是本次研究环境。运行需要兼容的 NVIDIA 驱动，不要求本地编译 Toolkit。`model.device` 支持 `auto/cpu/cuda`：auto 在 CUDA 不可见或矩阵运算/反向预检失败时记录原因并回退 CPU；cpu 强制 CPU；显式 cuda 不可用会失败。首次 CUDA 预检前统一设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，也允许显式 `:16:8`；非法值会报错，CUDA 已初始化后才发现缺失配置时要求重启进程，日志记录实际值。训练时的 OOM 等异常传播并保留失败档案，不自动续跑 CPU；重新尝试必须使用新运行目录和记录的种子。

依赖文件按用途分开：

| 文件 | 用途与状态 |
|---|---|
| `upstream/requirements.txt` | 固定提交的完整原始清单；除 torch 外上游未锁版本 |
| `requirements-cpu.lock.txt` | 当前实装验证后的核心运行闭包，全部固定版本 |
| `requirements-cuda.lock.txt` | 原 2.5.0+cu121 环境锁，非当前 RTX 5060 研究环境 |
| `requirements-blackwell.lock.txt` | Windows Python 3.11 / 2.7.1+cu128，已实际用于 RTX 5060 训练 |
| `requirements-dev.lock.txt` | pytest、ruff 与上游对照需要的 einops |
| `requirements-upstream-optional.lock.txt` | 当前实装过的上游 CPU 研究工具，包含训练框架、绘图、跟踪与 CPU ONNX；核心运行不需要 |
| `requirements-onnx-gpu.txt`、`requirements-onnx-gpu.lock.txt` | 可选 GPU ONNX 工具及完整实装锁，禁止作为 CPU 的必需依赖 |
| `dependency_provenance.json` | 原始来源、分组、版本调整与安装/计算验证状态 |

补充锁是 2026-10-15 的实装环境，**不是原论文完整环境**。GPU ONNX 采用实际存在的 `onnxruntime-gpu==1.20.0`；CPU ONNX 为 `1.20.1`。未实现 ONNX 导出/推理路线，本轮采用 PyTorch 二进制与平台 JSON 文本权重。

PyTorch wheel 通过限定到 torch 包的 find-links 提供，其余包从常规 Python 索引安装；避免 uv 在 PyTorch 索引中优先查找普通依赖的问题。按上述命令创建独立环境后运行测试和验收，不依赖仓库中已经存在某个 `.venv-tlob`；CPU 工程验收不代表真实数据预测有效或 GPU 可用。

## 数据、训练、档案与复跑

三个日常入口如下；本地训练和本地推理均支持 CPU/GPU，设备设置决定实际使用的硬件：

| 操作 | 显式 Python 文件 | 调用方式 |
|---|---|---|
| 本地训练 | `modified_tlob/train.py` | `python -m modified_tlob.train <训练配置.json>` |
| 本地只推理 | `modified_tlob/infer.py` | `python -m modified_tlob.infer --weights ... --bars ... --keys ... --output-root ...` |
| 组装提交材料 | `modified_tlob/package.py` | `python -m modified_tlob.package <训练运行目录> <新提交目录>` |

训练入口复用统一 `jinyi run` 流程，会训练、验证并存档；本地推理入口只加载权重，不调用 fit，不读取标签，不计算需要标签的评价指标。比赛平台的推理入口仍是 `submission.py` 中的 `main(datasources, start_date, end_date)`。

`configs/modified_tlob_cpu.json` 固定三档十二列：每档按 `ask_price/ask_volume/bid_price/bid_volume` 排列。初始主干为 hidden_dim=40、四个双轴块、一头、128 条历史、正弦编码。改变五档时必须使用 `schema.lob_columns(5)` 的二十列顺序。字段缺失、错序、错误维度或非法配置均失败。掩码只用于覆盖检查与缺历史回退；不拼入主干输入。

共享 `jinyiquant.data.preprocess_bars/build_batch` 完成缺失哨兵、有限前向填充、补零和因果窗口。训练只选达到 min_history 的样本；推理不足 min_history 的窗口返回训练样本标签均值。BiN 仅使用当前历史窗口的统计，无跨样本归一化拟合。下载价格单位为分、scale=100；平台行情为元、scale=1；两端最终都是元。平台单位转换明确保存在提交配置，缺失策略与字段/窗口参数随权重保存。

下面保留 `数据/` 下的 CPU 工程示例；当前7月 CUDA 对照使用 `data/research-july-20261015/bars/` 行情及配置中显式指定的官方标签、股票池和映射，来源与校验值随运行保存：

```powershell
$env:PYTHONPATH = 'src;.'
$tlobPython = '.venv-tlob/Scripts/python.exe'
& $tlobPython -m jinyiquant validate-config configs/modified_tlob_cpu.json
& $tlobPython -m jinyiquant run configs/modified_tlob_cpu.json
# 等价的显式训练模块入口：
& $tlobPython -m modified_tlob.train configs/modified_tlob_cpu.json
& $tlobPython -m jinyiquant rerun <运行目录>
& $tlobPython -m jinyiquant check <运行目录>
```

配置使用 2023-08-01 训练、08-02 研究验证，08-03 最终区间不读取；预算 300 秒/4096 MiB，epochs=2。这是工程试跑预算，不是已通过的真实数据性能/耗时承诺。正式训练需要由研究方向明确的数据区间和预算，不应把示例工程配置当成最优模型。

运行独立保存 `manifest.json`、`events.jsonl`、`config.json`、`source.zip`、数据版本、覆盖/边界检查、`metrics.json`、`daily.csv`、预测与提交三列、`report.txt`。权重为 `model.pt` 和 **`modified_tlob_weights.json`**；二者均保存结构、字段顺序、预处理口径、种子、训练均值及权重。二进制使用 `map_location`，文本按 dtype/shape 重建并移动到所选设备。模型更新、损失、学习率、验证 MSE、耗时、吞吐、CPU/GPU 资源按配置频率记录。验证只记录、不挑选检查点；保存达到预定 epochs 或 max_steps 时的最后权重。

误差容差预先固定 `atol=1e-6, rtol=1e-5`，包括保存加载、复跑及跨设备检查；本机 CPU/GPU 保存加载已通过。失败/中断保留堆栈与恢复状态，配置和结果不覆盖历史。所有成功、失败和预算超限的 run 进入 SQLite/JSONL 索引；完整目录与匹配数据应持久归档。缓存可重建，不能用缓存替代复现材料；没有自动清理最佳模型或历史档案的策略。

当前训练对照从冻结计划启动：

```powershell
$env:PYTHONPATH = 'src;.'
& .venv-tlob-cuda/Scripts/python.exe scripts/run_training_plan.py configs/training_step1_plan.json
```

| 研究选项 | 口径 |
|---|---|
| `model.target_scaling=none/train_std` | 仅以有效训练标签拟合尺度；预测还原为原始收益尺度，原始 MSE 按未加权收益误差计算 |
| `model.max_steps`、`data.training_sample_times` | 限制更新数、选择训练时点；评价仍使用固定的 `data.sample_times` |
| `model.time_loss_weighting=none/train_variance` | 仅用训练标签的分时方差确定损失权重，推理不用权重 |
| `model.book_representation=raw/relative_mid_bps` | 原价格或相对窗口末端中间价的基点表示 |
| `model.trade_features=zero/log1p` | 15列输入须配置三档盘口加 `volume/amount/deal_number`，以同维度的补零/对数成交特征作对照 |
| 离线平滑 | `scripts/compare_signal_smoothing.py <plan.json> --output <新目录>`，固定 α=0.5，当前截面标准化后按股逐日因果 EWMA，再还原尺度 |

本轮五步筛选及按预定门槛触发的复验已完成，结果与保留/回退依据见[结构化研究记录](training_progression_20261015.json)及[最终研究报告](../runs/training-progression/final-report-20261015/summary_zh.txt)。[训练时点配对图](../runs/training-progression/figures/step3c_training_times_paired.png)与[训练轮数对照图](../runs/training-progression/figures/step1_epochs_validation_mse.png)分别展示各自对照，不把不同实验串成一条提升曲线。模型、数据与完整运行档案留在本地。

研究主候选采用十个训练日、全部8时点、原始12列盘口、`target_scaling=train_std`、`time_loss_weighting=none`，不做信号平滑。本地参考固定使用最新研究验证窗（2023-07-25至07-28）的 seed 11，位于[本地研究参考包](../dist/local-research-reference-20261015-raw12-all8-s11-july25/)，并非正式冠军。

首轮2→12轮对照已完整执行；扩展数据后的十日/八时点实验固定1500次更新、batch_size=32，约为0.6个 epoch，不等于在扩展数据上完整训练12轮。时点扩展对照两臂均评价全部8时点。研究证据仅来自8个研究日，09:30开盘风险仍未解决；最终检查区间未打开，不能据本轮研究宣称官方成绩或完整区间稳定性。

当前配方另完成1500/5000/10000步训练时长对照（约0.6/2/4轮），两种长预算均未通过初筛，继续保留1500步研究参考。详情见[结构化记录](training_duration_20261015.json)、[本地报告](../runs/training-duration-20261015/screen-analysis/summary.txt)和[学习曲线](../runs/training-duration-20261015/screen-analysis/training_duration.png)。`logging.learning_curve_every_steps` 可按更新步数记录完整验证指标；观察过程保存并恢复随机数状态与训练模式，不改变优化轨迹，中途检查点只用于诊断。

生成合成数据并走同一入口：

```powershell
& $tlobPython scripts/tlob_fixture.py --output data/tlob-demo-001
& $tlobPython -m jinyiquant run data/tlob-demo-001/config.json
& $tlobPython scripts/tlob_acceptance.py --output runs/tlob-acceptance-002
```

验收脚本预先写 plan，运行 CPU 训练、保存、加载、预测、评价、复跑，组装提交包，启动无仓库导入路径且禁网的独立进程执行 Notebook。它用严格拒绝标签查询的本地 DAI 替身，报告为 `acceptance.json/package_smoke.json`；不冒充线上验收。合成权重只用于工程检查，不能证明真实预测质量。

## 本地只推理：infer.py

先完成一次训练，使用其运行目录中的 `model.pt` 或 `modified_tlob_weights.json`。PowerShell 从仓库根目录执行：

```powershell
$projectRoot = (Get-Location).Path
$env:PYTHONPATH = "$projectRoot\src;$projectRoot"
$tlobPython = "$projectRoot\.venv-tlob\Scripts\python.exe"
$trainingRun = '<已训练运行目录的绝对路径>'

& $tlobPython -m modified_tlob.infer `
  --weights "$trainingRun\model.pt" `
  --bars "$projectRoot\数据\cpt_jyc_2026_e2e_bar1m" `
  --keys "$projectRoot\数据\cpt_jyc_2026_instruments(1).parquet" `
  --instrument-map "$projectRoot\数据\cpt_jyc_2026_stock_bar1m_instrument_mapping.csv" `
  --start-date '2023-08-02 00:00:00' `
  --end-date '2023-08-03 00:00:00' `
  --device cpu --min-coverage 0.99 `
  --output-root "$projectRoot\runs\local-inference"
```

也可以直接执行文件，参数相同：`& $tlobPython "$projectRoot\modified_tlob\infer.py" --help`。模块方式依赖上面的 PYTHONPATH；直接执行文件会按源码位置定位公共包。无需修改 Notebook。CLI 成功时输出 JSON，给出此次运行目录和预测文件的实际路径；失败返回非零状态并保存失败档案。

参数说明：

| 参数 | 含义 |
|---|---|
| `--weights` | 必填，已训练的 `.pt` 或 JSON 文本权重；不存在则失败，不会训练兜底 |
| `--bars` | 必填，原始行情文件或月度分片目录；复用项目 Parquet/Feather/CSV/RAR 读取器 |
| `--keys` | 必填，独立的预测股票池/键文件，包含 date/instrument；不要按答案是否存在筛选键 |
| `--output-root` | 必填，每次在此目录下创建唯一 run_id，不覆盖历史预测 |
| `--device` | auto/cpu/cuda，默认 auto；GPU 需使用 CUDA 依赖环境 |
| `--instrument-map` | 行情仅含 instrument_id 时提供官方一对一映射；已有 instrument 可省略 |
| `--start-date / --end-date` | 可选，上海时间无时区，筛选预测键的半开区间 `[start, end)`；行情按保存的 history_days 补取历史 |
| `--price-scale` | 默认使用权重内保存的原始价格除数；输入为分用100、元用1。覆盖值进入日志，不更改权重 |
| `--min-coverage` | 每个截面的最小有效历史覆盖率，默认0.6；工程严格配置示例用0.99 |

预测键必须位于保存模型约定的采样时点、精确到分钟且无重复；不合法时失败。输出为新运行目录下的 `predictions.parquet`，严格三列 `date/instrument/factor`。同时保存 manifest、events、有效配置、数据校验、覆盖、检查、源码快照和设备信息。字段顺序、窗口长度、缺失处理和训练均值回退全部从权重恢复，不在推理数据上重新拟合。此入口不自动控制最终检查区间查看次数；日常研究请明确指定允许的开发区间。

CPU/GPU 使用同一文件。GPU 推理时将解释器改为 CUDA 环境的 Python，并设置 `--device cuda`；auto 可按预检结果回退 CPU，显式 cuda 不可用会报错。`modified_tlob_cpu.json` 中的设备只控制训练；本地只推理以这里的 `--device` 为准。

## 生成并上传提交包

从通过验收的训练目录生成新包（源码改变后必须先重新训练/复跑，打包器拒绝源码或权重校验不符）：

```powershell
& $tlobPython -m modified_tlob.package <运行目录> dist/modified-tlob-upload-001
```

上传 `dist/modified-tlob-upload-001` **内的文件与两个包目录**，保持同级结构；在该目录中打开唯一的 `inference.ipynb`。平台环境预先提供 Python 3.11、匹配的 PyTorch、核心依赖和 `dai`；Notebook 不执行 pip、不下载文件。缺依赖、模块、权重或校验失败均直接报错，不训练兜底。

以下为默认打包器生成的标准包清单，共 34 个文件。本轮本地研究参考包另归档 `requirements-blackwell.lock.txt`；实际文件及校验值以各包的 `package_manifest.json` 为准：

| 相对路径 | 用途 | 必需性 |
|---|---|---|
| `inference.ipynb` | 唯一推理 Notebook，定义可调用 main | 公榜必需 |
| `modified_tlob_weights.json` | 已训练的文本权重与完整预处理参数 | 公榜必需 |
| `submission_config.json` | 设备、平台单位、覆盖/时间预算、权重 SHA256 | 公榜必需 |
| `training_config.json` | 冻结开发切分、训练预算与可移植路径 | 私榜重训必需 |
| `package_manifest.json` | 所有上传文件 SHA256、来源 run、未验收状态 | 追溯必需 |
| `requirements-cpu.lock.txt` | CPU 环境准备；Notebook 不自动安装 | 环境声明 |
| `requirements-cuda.lock.txt` | CUDA 环境准备，运行时二选一 | 环境声明 |
| `modified_tlob/__init__.py` | 包标识 | 必需 |
| `modified_tlob/model.py` | 双轴主干、位置编码与回归网络 | 必需 |
| `modified_tlob/normalization.py` | BiN 与必要保护 | 必需 |
| `modified_tlob/schema.py` | 三/五档输入契约 | 必需 |
| `modified_tlob/device.py` | 设备预检与资源状态 | 必需 |
| `modified_tlob/adapter.py` | 训练、批次预测与权重加载 | 必需 |
| `modified_tlob/submission.py` | 平台数据查询及只推理 main | 必需 |
| `modified_tlob/train.py` | 隔离的显式 train_and_save | 私榜重训必需 |
| `modified_tlob/infer.py` | 独立本地文件推理入口；平台 main 不调用它 | 本地使用，随包保留 |
| `modified_tlob/provenance.json` | 固定论文/源码与变更清单 | 来源声明必需 |
| `modified_tlob/LICENSE` | 上游许可 | 必需 |
| `jinyiquant/__init__.py`、`jinyiquant/__main__.py` | 工程包和统一 CLI 入口 | 必需 |
| `jinyiquant/config.py`、`jinyiquant/io.py` | 配置/切分与 JSON/校验 | 必需 |
| `jinyiquant/data.py` | 共享数据处理、窗口和边界 | 必需 |
| `jinyiquant/tracking.py` | 统一日志和源代码/运行档案 | 必需 |
| `jinyiquant/runner.py`、`jinyiquant/checks.py` | 训练调度、独立预算监控与验收 | 私榜重训必需 |
| `jinyiquant/models.py`、`jinyiquant/evaluation.py` | 模型创建与本地评价 | 私榜重训必需 |
| `jinyiquant/cli.py`、`jinyiquant/preparation.py`、`jinyiquant/demo.py` | 完整 CLI 的依赖闭包 | 私榜重训必需 |
| `jinyiquant/experiments.py`、`jinyiquant/comparison.py`、`jinyiquant/protocol.py` | CLI 的复跑/对照依赖闭包 | 私榜重训必需 |

不要上传源码参照 `upstream/`、测试、scripts、开发用依赖/记录、原始数据、缓存、历史 runs、二进制 `model.pt`、其它 Notebook 或 `.venv`。打包器只复制白名单；运行过程生成的 inference_runs/training_runs 也不属于初始提交材料。

执行 Notebook 所有代码单元后，平台调用：

```python
factor_data = main(
    {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"},
    start_date, end_date,
)
```

`start_date/end_date` 为上海时间、无时区，平台入口对末端使用包含边界，训练/验证内部仍为半开区间。仅查询配置的历史行情与当时股票池，返回精确且仅含 `date/instrument/factor` 的 DataFrame；不读 vwap/target，不在注入区间训练，不借答案选择预测键。历史不足时按已记录策略回退，并按截面检查覆盖。加载路径由上传模块的位置解析，无本机绝对路径依赖。

私榜若按官方流程重训，可显式调用：

```python
from modified_tlob.train import train_and_save
weights_path = train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"})
```

它只读冻结训练/研究验证区间，保存独立完整档案并返回新文本权重路径，不覆盖原提交权重；用于私榜的替换/冻结组织方式应由实际平台流程验收。公榜 main 不调用它。

## 尚未完成的外部验收

本地 CPU 训练/复跑、上游中间表示对照、三档/五档梯度、常数/缺失窗口、因果性、错误设备、文本权重与断网包检查均有脚本或行为测试。原 Python 3.13 基线环境也独立验证。CI 增加可选 CPU 后端检查，尚未执行远端 CI。

真实7月数据与 RTX 5060 CUDA 训练、CPU/GPU 保存加载已完成本地检查，证据保留在 `runs/training-progression/` 和相应验收档案。官方平台上传运行、全区间 ≤3 小时、官方评价与最终检查仍**未验收**；当前短周期研究不能代替这些验收。本轮五步研究的保留/回退依据见[结构化研究记录](training_progression_20261015.json)，原始参考与全部候选档案均保留。
