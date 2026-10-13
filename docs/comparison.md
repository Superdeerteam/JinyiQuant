# 对照策略与结论

`jinyi suite plan.json` 的 plan 包含 baseline_config、candidate_config、seeds、periods、calibration_report、policy 六个字段。前三个路径字段相对于plan所在目录；periods 是 {start,end} 的数组。候选只改变 model 配置；数据、预处理、切分、预算和评价必须相同。

policy 必须预先声明以下所有字段，拒绝未知字段：

| 字段 | 意义 |
|---|---|
| primary_metric | 如 rank_ic_mean；仅接受能从日结果重算的已实现指标 |
| direction | maximize / minimize |
| min_improvement | 主指标有实际意义的绝对改善量，不能事后改 |
| max_metric_regressions | 非空次要指标字典，值为允许退化绝对量；也可 {max_regression,direction} |
| max_wall_ratio / max_memory_ratio | 每个配对中候选/基线成本上限 |
| min_seeds / min_periods | 至少2个种子、至少2个不重叠时段 |
| bootstrap_samples / block_days | 至少100次、连续交易日区块长度；每时段至少2个完整区块 |
| confidence / seed | 区块bootstrap置信度与随机种子 |
| threshold_rationale | 引用校准结果，解释实际阈值与代价限额 |

完整可运行的工程示例由 `scripts/acceptance.py` 自动生成。它只对相同模型进行零改善对照，故不能照搬其阈值用于研究晋升。

指标和不确定性：各截面统计先按交易日等权平均，再从日序列计算均值、IR和夏普；日夏普年化默认252。多空是前向标签的分组差值，并非模拟账户净值。extreme_return 为最差5%交易日多空值平均，属于明确的本地压力近似。换手率为相邻截面两侧权重变化的平均单边换手，首截面未定义；并列分数共享边界组权重。常数序列的相关性、无波动的IR/夏普记 null，不用0伪装。metrics 中保存全部具体约定。

comparison 从日结果重算指标，验证数值、样本/标签签名和产物校验值；跨种子同步抽区块防止人为扩大样本量。输出总体置信区间、逐配对变化、基线按种子/时段波动、成本及各门槛。通过建议不会自动修改“最佳模型”。

适用边界：固定CPU闭式基线没有随机优化，因此种子差异可能为0；不能据此推断未来神经网络的种子稳定性。区块长度/时段选择仍须依据研究对象预先声明；反复筛选会引入选择偏差，当前区间仅对所列尝试条件化，不是对全部搜索过程的显著性保证。
