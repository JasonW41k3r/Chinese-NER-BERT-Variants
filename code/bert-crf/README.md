# BERT-CRF 中文命名实体识别（实验说明）

## 1. 简介
本实验在 **BERT + CRF** 框架上进行中文 NER 训练与评估，标签按 **每字一个标签**（BIO/IOBES 等）给出。

## 2. 数据准备
将数据放在 `./data/` 目录：
```
data/
  train.txt
  dev.txt
  test.txt
```
- 每行：`字 空格 标签`
- 句子之间以**空行**分隔。

## 3. 环境依赖
```bash
pip install transformers==4.44.2 torchcrf==1.1.0 seqeval==1.2.2
```

## 4. 训练与评估（示例命令，RTX 2050 友好）
```bash
python train_bert_crf.py \
  --data_dir ./data \
  --out_dir ./run \
  --pretrained bert-base-chinese \
  --batch_size 16 --eval_batch_size 16 \
  --max_len 160 --epochs 5 --lr 2e-5
```
脚本会在每个 epoch 结束时评估 `dev`，并将最优模型保存到 `run/best/`，最后在 `test` 上输出：
- `run/test_report.txt`：按实体类型的指标
- `run/test_metrics.json`：总体 P/R/F1

## 5. 常用超参数
- `--batch_size`: 8~32（显存受限用 8 或 16）
- `--max_len`: 常用 160~256
- `--lr`: 1e-5 ~ 3e-5
- `--epochs`: 3~10

## 6. 常见问题（FAQ）
- **mask of the first timestep must all be on**：通常是 mask/对齐问题，确保每个样本第一位 mask=1（脚本已处理）。
- **UndefinedMetricWarning（某些标签无预测）**：早期 epoch 正常；可训练更多 epoch 或略调学习率。
- **显存不足**：减小 `--batch_size` 或 `--max_len`。

## 7. 结果产物
```
run/
  best/                # 最优 checkpoint（可用于推理）
  dev_report_epoch*.txt
  test_report.txt
  test_metrics.json
```
