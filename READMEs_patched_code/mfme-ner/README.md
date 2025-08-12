# MFME-NER（Multi-Feature Memory Encoding for NER）实验说明

## 1. 简介
在 BERT 字向量基础上，融合多种**记忆特征**：Lexicon 候选、Bigram、POS、字符类型，使用注意力读出 + CRF 解码。已内置数值稳定与安全 softmax。

## 2. 数据与特征构建
```bash
python build_features.py --data_dir ./data --out_dir ./feats --min_word_len 2 --min_freq 2
```
生成：
- `feats/lexicon.json`
- `feats/bigram.json`
- `feats/pos_vocab.json`

## 3. 依赖
```bash
pip install transformers==4.44.2 torchcrf==1.1.0 seqeval==1.2.2 jieba
```

## 4. 训练与评估（示例）
```bash
python train_mfme.py \
  --data_dir ./data --out_dir ./run \
  --pretrained bert-base-chinese \
  --lexicon_json ./feats/lexicon.json \
  --bigram_json ./feats/bigram.json \
  --pos_json ./feats/pos_vocab.json \
  --batch_size 8 --eval_batch_size 16 \
  --max_len 160 --topk 5 \
  --lex_dim 100 --bigram_dim 50 --pos_dim 16 --type_dim 8 --mem_hidden 128 \
  --epochs 5 --lr 1e-5
```
（首次建议 `--lr 1e-5` 更稳）

## 5. 常见问题
- **loss=NaN / 爆炸**：本仓库已加入安全 softmax、logits 限幅与 CRF 转移 clamp；若仍异常，降 `--lr`，或把 `--lex_dim 64 --mem_hidden 96`。

## 6. 产物
同前：`run/best/`、`test_report.txt`、`test_metrics.json`。
