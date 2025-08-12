# SpanKL 中文命名实体识别（实验说明）

## 1. 简介
SpanKL 基于 **span 表达**进行实体识别，结合知识/特征，对候选跨度进行打分与解码（实现为稳定简化版）。

## 2. 数据
```
data/
  train.txt  dev.txt  test.txt
```
每行 `字 空格 标签`，句间空行。

## 3. 依赖
```bash
pip install transformers==4.44.2 seqeval==1.2.2
```

## 4. 训练与评估（示例）
```bash
python train_spankl.py \
  --data_dir ./data --out_dir ./run \
  --pretrained bert-base-chinese \
  --batch_size 8 --eval_batch_size 16 \
  --max_len 160 --epochs 5 --lr 2e-5
```
脚本会自动在 `dev` 上选最优模型并于 `test` 上评估。

## 5. 常见问题
- **显存不足**：降低 `--batch_size` 或 `--max_len`。
- **分类不收敛**：降低学习率至 `1e-5`，或增大 epoch。

## 6. 产物
`run/best/`、`test_report.txt`、`test_metrics.json`。
