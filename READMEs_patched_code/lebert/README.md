# LEBERT 中文命名实体识别（实验说明）

## 1. 简介
LEBERT 在 BERT 的字向量上融合**词级特征**（Lexicon），再接 CRF 进行序列标注。

## 2. 数据与词表
```
data/
  train.txt  dev.txt  test.txt
```
构建词典（基于训练集，搜索模式）：
```bash
python build_lexicon.py --data_dir ./data --out_dir ./feats --min_word_len 2 --min_freq 2
```
生成：`feats/lexicon.json`。

## 3. 依赖
```bash
pip install transformers==4.44.2 torchcrf==1.1.0 seqeval==1.2.2 jieba
```

## 4. 训练与评估（示例）
```bash
python train_lebert.py \
  --data_dir ./data \
  --out_dir ./run \
  --pretrained bert-base-chinese \
  --lexicon_json ./feats/lexicon.json \
  --batch_size 8 --eval_batch_size 16 \
  --epochs 5 --max_len 160 --topk 5 --lex_dim 100
```
输出与目录结构与 BERT-CRF 一致：`run/best/`、`test_report.txt`、`test_metrics.json`。

## 5. 建议超参
- `--topk`: 每个字保留的候选词数，常用 5~8
- `--lex_dim`: 64~128；显存紧可降至 64
- 其他同 BERT-CRF

## 6. 常见问题
- **loss 数值异常/爆炸**：将学习率降为 `1e-5`；或减小 `--lex_dim`、`--topk`。
- **加载最优模型报缺少参数**：确保 `--lexicon_json` 与训练时一致；或使用脚本中 `from_pretrained(best_path, lexicon_size=...)` 的方式加载。
