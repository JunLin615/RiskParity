# 训练入口配置文件说明：`train_run_config_example.json`

该 JSON 文件用于启动 `run_train_ranker.py`，控制一次完整训练实验的输入数据、训练/验证切分、Dataset 采样方式、模型规模、优化器参数、损失函数、checkpoint 保存方式等。

运行方式：

```bash
python run_train_ranker.py --config train_run_config_example.json
```

每次运行后，训练结果会保存到：

```text
checkpoint_dir / 时间戳文件夹 /
```

例如：

```text
checkpoints/dual_transformer_ranker/
  20260503_153012/
    train_config.json
    model_config.json
    dataset_summary.json
    launcher_config.json
    input_config.json
    history.csv
    best.pt
    last.pt
    run_summary.json
```

---

## 1. 完整配置示例

```json
{
  "bundle_path": "data/cache/stage1_factor_label_bundle.pkl",
  "checkpoint_dir": "checkpoints/dual_transformer_ranker",
  "run_name": null,

  "train_end": "20231231",
  "valid_end": "20241231",
  "valid_ratio": 0.2,

  "sample_size": 512,
  "seq_len": 128,
  "samples_per_date": 1,
  "label_name": "label_ret_t1_t6",
  "label_valid_name": "label_valid_t1_t6",
  "target_mode": "rank_pct",
  "candidate_pool_size": null,
  "allow_smaller_sample": false,
  "require_full_history": true,

  "model_preset": "small",

  "max_epochs": 10,
  "batch_size": 1,
  "num_workers": 0,
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "grad_clip_norm": 1.0,
  "loss_type": "spearman",
  "tau_start": 1.0,
  "tau_end": 0.1,
  "tau_decay_epochs": 50,
  "topk_metric_k": 20,
  "device": "auto",
  "seed": 42,
  "early_stopping_patience": 10,
  "log_every_steps": 50,
  "use_amp": true
}
```

---

# 2. 数据与结果保存相关参数

## `bundle_path`

```json
"bundle_path": "data/cache/stage1_factor_label_bundle.pkl"
```

含义：  
训练所使用的特征与标签缓存文件路径。

这个文件通常由 `factor_pipeline.py` 生成，内部包含：

```text
feature_panel
label_panel
factor_names
label_names
ts_factor_names
scalar_factor_names
metadata
```

推荐：

```text
每次因子、标签、股票池过滤逻辑发生变化，都重新生成新的 bundle。
```

---

## `checkpoint_dir`

```json
"checkpoint_dir": "checkpoints/dual_transformer_ranker"
```

含义：  
训练实验的根目录。

实际保存时，每次训练会自动创建一个时间戳子文件夹，例如：

```text
checkpoints/dual_transformer_ranker/20260503_153012/
```

该目录下保存：

```text
best.pt
last.pt
history.csv
train_config.json
model_config.json
dataset_summary.json
launcher_config.json
run_summary.json
```

---

## `run_name`

```json
"run_name": null
```

含义：  
手动指定本次实验文件夹名称。

当为 `null` 时，系统自动使用时间戳命名：

```text
20260503_153012
```

也可以手动指定：

```json
"run_name": "small_lr1e-4_rankpct_v1"
```

则保存目录为：

```text
checkpoints/dual_transformer_ranker/small_lr1e-4_rankpct_v1/
```

建议：  
正式实验可以保留 `null`，让系统自动生成时间戳；重要实验可以手动命名。

---

# 3. 训练集 / 验证集切分参数

## `train_end`

```json
"train_end": "20231231"
```

含义：  
训练集结束日期。

所有信号日期：

```text
trade_date <= train_end
```

会被划入训练集。

推荐：  
使用时间切分，不要随机切分日期，避免未来信息泄露。

---

## `valid_end`

```json
"valid_end": "20241231"
```

含义：  
验证集结束日期。

当设置了 `train_end` 和 `valid_end` 时：

```text
train: trade_date <= train_end
valid: train_end < trade_date <= valid_end
```

示例：

```text
训练集：2020-01-01 到 2023-12-31
验证集：2024-01-01 到 2024-12-31
```

---

## `valid_ratio`

```json
"valid_ratio": 0.2
```

含义：  
当不设置 `train_end` 时，使用最后一部分日期作为验证集。

例如：

```text
valid_ratio = 0.2
```

表示最后 20% 的日期作为验证集。

注意：  
如果已经设置了 `train_end`，则优先使用时间切分，`valid_ratio` 的作用会减弱。

---

# 4. Dataset 采样参数

## `sample_size`

```json
"sample_size": 512
```

含义：  
每个训练样本中随机抽取的股票数量，即截面大小 `N`。

对应模型输入：

```text
x_ts:     [N, F_ts, T]
x_scalar: [N, F_scalar]
y:        [N]
```

当前策略设计中：

```text
N = 512
```

推荐：

```text
4060 8G 本地调试：256 或 512
A100 / 大显存：512
```

如果显存不足，优先降低：

```text
sample_size
model_dim
seq_len
```

---

## `seq_len`

```json
"seq_len": 128
```

含义：  
每只股票输入模型的历史时间长度 `T`。

对应：

```text
x_ts: [N, F_ts, T]
```

当前设计中：

```text
T = 128
```

推荐：

```text
初始训练：128
显存不足：64
更长周期研究：192 或 256
```

---

## `samples_per_date`

```json
"samples_per_date": 1
```

含义：  
每个交易日在一个 epoch 内生成多少个随机截面样本。

例如基础股票池有 2048 只股票，每次随机抽 512 只：

```text
samples_per_date = 1
```

表示每个日期每个 epoch 抽样一次。

```text
samples_per_date = 4
```

表示同一个日期每个 epoch 随机抽样 4 次。

作用：

```text
增加截面组合多样性
起到数据增强和正则化作用
降低模型死记股票组合的风险
```

建议：

```text
本地调试：1
正式训练：2 到 4
大算力：4 到 8
```

---

## `label_name`

```json
"label_name": "label_ret_t1_t6"
```

含义：  
训练使用的标签列名。

当前默认标签：

```text
label_ret_t1_t6[t] = log(close[t+6] / close[t+1])
```

对应策略逻辑：

```text
t 日生成信号
t+1 买入
t+6 卖出
```

可选常用标签：

```text
label_ret_t1_t6          使用复权收盘价计算的 t+1 到 t+6 对数收益
label_ret_exec_t1_t6     使用保守成交价计算的 t+1 到 t+6 对数收益
```

说明：  
第一阶段建议先用 `label_ret_t1_t6`。等回测框架稳定后，可以测试执行价标签。

---

## `label_valid_name`

```json
"label_valid_name": "label_valid_t1_t6"
```

含义：  
标签有效性掩码列名。

只有该字段为真，并且标签非空的股票，才会进入训练样本。

通常包含：

```text
未来 t+1 和 t+6 数据存在
股票在 t 日属于有效股票池
必要时 t+1 可买、t+6 可卖
```

如果不想使用有效性掩码，可以设为：

```json
"label_valid_name": null
```

但一般不建议关闭。

---

## `target_mode`

```json
"target_mode": "rank_pct"
```

含义：  
在每次随机抽样出的 512 只股票内部，如何把未来收益转换成训练目标。

可选：

```text
rank_pct        局部百分位排名，最低接近 1/N，最高为 1
rank_centered   2 * rank_pct - 1，范围约 [-1, 1]
rank            原始局部排名，1 到 N
zscore          抽样截面内的收益 z-score
raw             原始未来收益
```

当前推荐：

```text
rank_pct
```

原因：  
模型目标是截面排序，而不是精确预测收益率数值。`rank_pct` 对极端收益更稳。

---

## `candidate_pool_size`

```json
"candidate_pool_size": null
```

含义：  
限制每日候选股票池的最大数量。

例如：

```json
"candidate_pool_size": 2048
```

表示每天先从有效股票中取最多 2048 只，再从中随机抽 `sample_size` 只。

当前为 `null`，表示不额外限制，由上游 bundle 中的股票池决定。

推荐：  
如果上游 `factor_pipeline` 已经严格控制基础股票池，可以保持 `null`。  
如果想在 Dataset 层模拟“2048 基础池”，可以设置为 `2048`。

---

## `allow_smaller_sample`

```json
"allow_smaller_sample": false
```

含义：  
当某个交易日有效股票数少于 `sample_size` 时，是否允许使用更小的样本。

当前设置：

```text
false
```

表示如果某日有效股票不足 512，则该日不可用于训练。

推荐：  
正式训练保持 `false`，保证输入形状固定。

---

## `require_full_history`

```json
"require_full_history": true
```

含义：  
是否要求每个训练日期都具有完整的 `seq_len` 历史窗口。

例如：

```text
seq_len = 128
```

则每个信号日必须至少有过去 128 个交易日的因子数据。

推荐：

```text
正式训练：true
调试数据较短时：false
```

---

# 5. 模型结构参数

## `model_preset`

```json
"model_preset": "small"
```

含义：  
模型规模预设。

当前入口脚本支持：

```text
small
medium
base
```

### `small`

适合 RTX 4060 8G 本地测试：

```text
model_dim = 128
temporal_channels = 8
temporal_compressed_len = 16
factor_num_layers = 1
cross_num_layers = 1
```

### `medium`

中等规模：

```text
model_dim = 256
```

适合本地显存够用后进一步测试。

### `base`

原始设计规模：

```text
model_dim = 512
temporal_channels = 16
temporal_compressed_len = 32
```

更适合服务器或大显存 GPU。

推荐训练顺序：

```text
small → medium → base
```

不要一开始直接上 `base`，否则排查问题成本较高。

---

# 6. 训练参数

## `max_epochs`

```json
"max_epochs": 10
```

含义：  
最大训练轮数。

建议：

```text
流程测试：1 到 3
小模型初步观察：5 到 10
正式训练：30 到 100
```

---

## `batch_size`

```json
"batch_size": 1
```

含义：  
DataLoader 外层 batch size。

注意：  
这里的 `batch_size=1` 并不是只训练 1 只股票，而是一次训练 1 个完整截面样本。

一个样本内部已经包含：

```text
sample_size = 512 只股票
```

所以实际输入形状是：

```text
[B, N, F_ts, T]
```

当前推荐：

```text
batch_size = 1
```

原因：  
截面 attention 的显存消耗随 `N²` 增长，外层 batch size 不宜过大。

---

## `num_workers`

```json
"num_workers": 0
```

含义：  
DataLoader 多进程数量。

推荐：

```text
Windows / Notebook：0
Linux 服务器：2 到 8
```

如果遇到 DataLoader 卡死或多进程错误，先设为 0。

---

## `lr`

```json
"lr": 0.0001
```

含义：  
学习率。

推荐范围：

```text
1e-5 到 3e-4
```

初始建议：

```text
1e-4
```

如果 loss 不动，可以试：

```text
3e-4
```

如果训练非常不稳定，可以试：

```text
3e-5 或 1e-5
```

---

## `weight_decay`

```json
"weight_decay": 0.0001
```

含义：  
AdamW 权重衰减，控制过拟合。

推荐：

```text
1e-5 到 1e-3
```

当前：

```text
1e-4
```

是比较稳妥的初始值。

---

## `grad_clip_norm`

```json
"grad_clip_norm": 1.0
```

含义：  
梯度裁剪阈值。

作用：

```text
防止梯度爆炸
提高 Transformer 训练稳定性
```

推荐：

```text
1.0
```

如果训练很稳定，可以尝试关闭：

```json
"grad_clip_norm": null
```

---

# 7. 损失函数与温度退火

## `loss_type`

```json
"loss_type": "spearman"
```

含义：  
训练损失类型。

可选：

```text
spearman    soft-rank Spearman 风格排序损失
pearson     预测分数与目标 rank 的 Pearson 相关损失
pairwise    pairwise logistic 排序损失
```

当前推荐：

```text
spearman
```

---

## `tau_start`

```json
"tau_start": 1.0
```

含义：  
soft-rank 温度初始值。

温度越高：

```text
排序越平滑
梯度更稳定
但排序边界更模糊
```

---

## `tau_end`

```json
"tau_end": 0.1
```

含义：  
soft-rank 温度最终值。

温度越低：

```text
越接近硬排序
更强调精确排序
但训练可能更不稳定
```

---

## `tau_decay_epochs`

```json
"tau_decay_epochs": 50
```

含义：  
温度从 `tau_start` 衰减到 `tau_end` 所用的 epoch 数。

例如：

```text
tau_start = 1.0
tau_end = 0.1
tau_decay_epochs = 50
```

表示前 50 个 epoch 逐步从平滑排序过渡到更接近硬排序。

如果只训练 10 个 epoch，温度不会完全降到 0.1。

---

# 8. 指标与设备参数

## `topk_metric_k`

```json
"topk_metric_k": 20
```

含义：  
训练和验证时额外统计 Top-K 平均未来收益。

例如：

```text
topk_metric_k = 20
```

表示每个截面中，取模型预测分数最高的 20 只股票，计算其真实未来收益均值。

该指标不是训练损失，只是观察指标。

---

## `device`

```json
"device": "auto"
```

含义：  
训练设备。

可选：

```text
auto    自动选择 cuda 或 cpu
cuda    强制使用 GPU
cpu     强制使用 CPU
```

推荐：

```text
auto
```

---

## `seed`

```json
"seed": 42
```

含义：  
随机种子。

影响：

```text
随机抽样股票
模型初始化
训练过程随机性
```

建议每次实验记录 seed，便于复现。

---

## `early_stopping_patience`

```json
"early_stopping_patience": 10
```

含义：  
早停 patience。

如果连续 10 个 epoch 验证指标没有提升，则提前停止训练。

当前默认使用：

```text
valid_rank_ic
```

作为最优模型判断指标。

如果不想早停，可以设置：

```json
"early_stopping_patience": null
```

---

## `log_every_steps`

```json
"log_every_steps": 50
```

含义：  
每隔多少个训练 step 打印一次中间训练日志。

推荐：

```text
本地调试：10 到 50
正式训练：50 到 200
```

---

## `use_amp`

```json
"use_amp": true
```

含义：  
是否使用自动混合精度训练。

推荐：

```text
GPU 训练：true
CPU 训练：false
```

作用：

```text
降低显存占用
提高训练速度
```

对于 RTX 4060 8G，建议开启。

---

# 9. 第一轮推荐配置

RTX 4060 8G 上，第一轮建议不要追求最终效果，而是确认训练链路正常：

```json
{
  "model_preset": "small",
  "sample_size": 256,
  "seq_len": 128,
  "samples_per_date": 1,
  "max_epochs": 3,
  "batch_size": 1,
  "lr": 0.0001,
  "use_amp": true
}
```

确认以下几点：

```text
1. 能正常读取 bundle
2. 能正常构造 train / valid dataset
3. 能正常开始训练
4. loss 不是 NaN
5. valid_rank_ic 能正常计算
6. history.csv、best.pt、last.pt 能正常保存
```

确认流程稳定后，再逐步提高：

```text
sample_size: 256 → 512
samples_per_date: 1 → 2 → 4
model_preset: small → medium → base
max_epochs: 3 → 10 → 50
```

---

# 10. 实验记录建议

每次训练后，在实验日志中记录：

```text
实验时间：
run_dir：
bundle_path：

训练区间：
验证区间：

model_preset：
sample_size：
seq_len：
samples_per_date：
label_name：
target_mode：

max_epochs：
lr：
weight_decay：
loss_type：
tau_start：
tau_end：

best_epoch：
best valid_rank_ic：
best valid_ic：
best valid_topk_ret：
final train_loss：
final valid_loss：

现象：
问题：
下一步计划：
```
