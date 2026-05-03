# Dual-Transformer 截面排序模型训练流程极简说明

本文档用于记录训练流程、参数设置和训练结果。建议每次训练后在最后的“实验记录”部分追加一条记录，方便后续排查问题和对比模型。

---

## 1. 当前训练链路

```text
stock_data.py
→ factor_library.py
→ label_builder.py
→ factor_pipeline.py
→ rank_dataset.py
→ dual_transformer_model.py
→ rank_loss.py
→ train_ranker.py
```

各文件作用：

```text
factor_pipeline.py      生成 feature_panel / label_panel
rank_dataset.py         随机抽样 512 股票，构造 [N, F_ts, T] 和 [N, F_scalar]
dual_transformer_model.py  模型结构
rank_loss.py            排序损失与 RankIC / IC / TopK 指标
train_ranker.py         训练循环、验证、checkpoint
```

---

## 2. 生成或加载 bundle

### 2.1 从数据库生成

```python
import factor_pipeline as fp
from stock_data import create_stock_manager, load_tushare_token

TOKEN = load_tushare_token("tushare_token.txt")

manager = create_stock_manager(
    tushare_token=TOKEN,
    db_path="data/db/stock_data.db",
    default_start_date="20180101",
)

pipe_config = fp.FactorPipelineConfig(
    start_date="20200101",
    end_date="20241231",

    # 为滚动窗口预留历史，为 t+6 标签预留未来
    load_start_date="20190601",
    load_end_date="20250131",

    feature_adjust_type="qfq",
    label_adjust_type="qfq",
    execution_adjust_type="raw",

    standardize_features=True,
    winsorize_features=True,

    include_close_to_close_label=True,
    include_execution_label=True,
)

bundle = fp.build_bundle_from_manager(
    manager,
    config=pipe_config,
)

fp.save_bundle(bundle, "data/cache/stage1_factor_label_bundle.pkl")
```

### 2.2 直接加载

```python
import factor_pipeline as fp

bundle = fp.load_bundle("data/cache/stage1_factor_label_bundle.pkl")
```

检查：

```python
bundle.metadata
bundle.feature_panel.shape
bundle.label_panel.shape
```

---

## 3. 构建 Dataset

```python
import rank_dataset as rd
import train_ranker as tr

ds_config = rd.CrossSectionDatasetConfig(
    sample_size=512,
    seq_len=128,
    samples_per_date=4,

    label_name="label_ret_t1_t6",
    label_valid_name="label_valid_t1_t6",

    target_mode="rank_pct",
    require_full_history=True,
    return_tensors="torch",

    random_seed=42,
)
```

时间切分：

```python
train_ds, valid_ds = tr.build_datasets_from_bundle(
    bundle,
    ds_config=ds_config,
    train_end="20231231",
    valid_end="20241231",
)
```

检查：

```python
train_ds.summary()
valid_ds.summary()
```

---

## 4. 构建模型

第一版建议先用偏小模型确认训练流程：

```python
model = tr.build_model_from_bundle(
    bundle,
    seq_len=128,

    temporal_channels=8,
    temporal_compressed_len=16,
    model_dim=128,

    factor_num_layers=1,
    factor_num_heads=4,
    factor_ff_dim=256,

    cross_num_layers=1,
    cross_num_heads=4,
    cross_ff_dim=256,

    score_hidden_dim=128,
    dropout=0.1,
)
```

确认训练稳定后再用正式版本：

```python
model = tr.build_model_from_bundle(
    bundle,
    seq_len=128,

    temporal_channels=16,
    temporal_compressed_len=32,
    model_dim=512,

    factor_num_layers=1,
    factor_num_heads=8,
    factor_ff_dim=1024,

    cross_num_layers=1,
    cross_num_heads=8,
    cross_ff_dim=1024,

    score_hidden_dim=512,
    dropout=0.1,
)
```

---

## 5. 训练

```python
train_config = tr.TrainConfig(
    max_epochs=50,
    batch_size=1,

    lr=1e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,

    loss_type="spearman",

    tau_start=1.0,
    tau_end=0.1,
    tau_decay_epochs=50,

    topk_metric_k=20,

    device="auto",
    seed=42,

    checkpoint_dir="checkpoints/dual_transformer_ranker",
    save_best=True,
    save_last=True,

    early_stopping_patience=10,
    metric_for_best="valid_rank_ic",
    maximize_metric=True,
)
```

开始训练：

```python
history = tr.fit_model(
    model,
    train_ds,
    valid_ds,
    train_config,
)

history.to_csv("checkpoints/dual_transformer_ranker/history.csv", index=False)
```

---

## 6. 训练时重点看什么

优先看：

```text
valid_rank_ic
valid_ic
valid_topk_ret
train_loss / valid_loss 是否异常
```

一般来说：

```text
train_rank_ic 上升，但 valid_rank_ic 不上升：
    可能过拟合，降低模型规模、加 dropout、减少 samples_per_date 或加强正则。

train_loss 几乎不变：
    可能学习率太低、loss 温度不合适、标签/特征对齐有问题。

valid_topk_ret 和 valid_rank_ic 背离：
    可能模型整体排序相关性一般，但头部有效；也可能 TopK 样本太少，需要回测确认。

valid_rank_ic 为负：
    优先检查标签方向、收益计算区间、特征/标签日期是否错位。
```

---

## 7. 常用调参入口

### 数据集参数

```text
sample_size          默认 512
seq_len              默认 128
samples_per_date     默认 4，可试 1 / 2 / 4 / 8
target_mode          rank_pct / rank_centered / zscore / raw
label_name           label_ret_t1_t6 或 label_ret_exec_t1_t6
```

### 模型参数

```text
model_dim
temporal_channels
temporal_compressed_len
factor_num_layers
cross_num_layers
factor_num_heads
cross_num_heads
dropout
```

### 训练参数

```text
lr
weight_decay
loss_type
tau_start / tau_end / tau_decay_epochs
grad_clip_norm
early_stopping_patience
```

---

## 8. 训练入口示意

1. 先生成训练集 / bundle
python run_build_training_bundle.py --config build_and_train_config_example.json

它会读取：

stock_data.py 数据库
daily / daily_basic / moneyflow / eligibility / can_buy / can_sell

然后生成：

data/cache/stage1_factor_label_bundle.pkl
data/cache/stage1_factor_label_bundle.build_config.json
data/cache/stage1_factor_label_bundle.metadata.json
data/cache/stage1_factor_label_bundle.input_config.json

其中 stage1_factor_label_bundle.pkl 就是后续训练使用的训练数据缓存。

2. 再启动训练
python run_train_ranker.py --config build_and_train_config_example.json

它会读取：

data/cache/stage1_factor_label_bundle.pkl

然后创建：

checkpoints/dual_transformer_ranker/
  时间戳文件夹/
    train_config.json
    model_config.json
    dataset_summary.json
    launcher_config.json
    input_config.json
    history.csv
    best.pt
    last.pt
    run_summary.json

cmd中去checkpoints中对应当此训练的目录下，
tensorboard --logdir checkpoints/dual_transformer_ranker

tensorboard看训练监测。
## 8. 实验记录模板

每次训练后复制一份。

```text
实验编号：
日期：
数据范围：
训练集：
验证集：

label_name：
target_mode：
sample_size：
seq_len：
samples_per_date：

模型参数：
temporal_channels：
temporal_compressed_len：
model_dim：
factor_num_layers：
factor_num_heads：
cross_num_layers：
cross_num_heads：
dropout：

训练参数：
lr：
weight_decay：
loss_type：
tau_start：
tau_end：
max_epochs：
early_stopping_patience：

最好 epoch：
best valid_rank_ic：
best valid_ic：
best valid_topk_ret：
最终 train_loss：
最终 valid_loss：

现象：
问题：
下一步修改：
```

---

## 9. 当前建议的第一轮训练策略

先小模型：

```text
sample_size = 512
seq_len = 128
samples_per_date = 1 或 2
model_dim = 128
factor_num_layers = 1
cross_num_layers = 1
max_epochs = 5 到 10
```

目标不是追求收益，而是确认：

```text
1. 训练流程不报错
2. loss 能正常下降或波动合理
3. RankIC 不是长期 NaN
4. valid_topk_ret 可以正常计算
```

小模型跑通后，再逐步增加：

```text
samples_per_date
model_dim
cross_num_layers
训练 epoch
```
