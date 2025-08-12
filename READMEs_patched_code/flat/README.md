# FLAT（Flat-Lattice Transformer）实验说明

## 1. 简介
基于 BERT 的字表示，构造**扁平词格**（字节点 + 词节点），使用带相对位置偏置的多头自注意力编码，最终在**字序列**上用 CRF 解码。已处理数值稳定与索引类型问题。

## 2. 数据与词表
```bash
python build_lexicon.py --data_dir ./data --out_dir ./feats --min_word_len 2 --min_freq 2 --max_words 50000
```
生成：`feats/lexicon.json`。

## 3. 依赖
```bash
pip install transformers==4.44.2 torchcrf==1.1.0 seqeval==1.2.2 jieba
```

## 4. 训练与评估（示例）
```bash
python train_flat.py \
  --data_dir ./data --out_dir ./run \
  --pretrained bert-base-chinese \
  --lexicon_json ./feats/lexicon.json \
  --batch_size 8 --eval_batch_size 16 \
  --max_len 160 --max_words 50 \
  --layers 2 --heads 8 --ff 1024 \
  --lr 1e-5
```
若显存充裕，可将 `--batch_size` 提至 16；若显存紧张，把 `--max_words` 降到 30。

## 5. 常见问题
- **embedding 索引类型错误（Float/Long）**：已在代码中将相对位置索引强制 `long`；如自改请保持该约束。
- **训练不稳定**：可降 `--heads` 到 4，或降低 `--lr`。

## 6. 产物
同前：`run/best/`、`test_report.txt`、`test_metrics.json`。
