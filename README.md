
# Chinese-NER-BERT-Variants
**基于BERT的多模型中文命名实体识别实验实现**

## Introduction | 简介
This repository implements and benchmarks **five BERT-based Chinese Named Entity Recognition (NER) models**:

- **BERT-CRF**: Baseline model combining BERT and Conditional Random Fields.  
- **LEBERT**: Lexicon-Enhanced BERT integrating external lexicon information.  
- **SpanKL**: Span-based NER with Kullback–Leibler divergence optimization.  
- **MFME-NER**: Multi-Feature Memory Encoding for multi-source feature fusion.  
- **FLAT**: Flat-Lattice Transformer for character-word joint modeling.  

本仓库实现并对比了 **5 种基于 BERT 的中文命名实体识别模型**，涵盖特征构造、训练、评估与性能对比，并在 **RTX 2050 GPU 环境**下进行了优化。

---

## Repository Structure | 仓库结构
```

code/
│── bert-crf/       # BERT-CRF model implementation
│── lebert/         # LEBERT model implementation
│── spankl/         # SpanKL model implementation
│── mfme-ner/       # MFME-NER model implementation
│── flat/           # FLAT model implementation
│── feats/          # External feature files (lexicon, bigram, POS, etc.)
│── data/           # Dataset files
│── utils/          # Utility scripts
│── requirements.txt
│── README.md

````

---

## Installation | 安装
```bash
# Create and activate environment
conda create -n ner-env python=3.10
conda activate ner-env

# Install dependencies
pip install -r requirements.txt
````

---

## Dataset | 数据集

The models are trained and evaluated on **Chinese NER datasets** (e.g., MSRA, Resume, Weibo).
数据集应放置在 `data/` 目录下，格式应包含：

```
train.json
dev.json
test.json
```

每个文件包含分词后的文本与标签。

---

## Usage | 使用方法

### 1. Train a model | 训练模型

Example: Train BERT-CRF

```bash
cd bert-crf
python train_bertcrf.py \
    --data_dir ../data \
    --out_dir ../run \
    --pretrained bert-base-chinese \
    --batch_size 16 \
    --eval_batch_size 16 \
    --max_len 160 \
    --epochs 5 \
    --lr 2e-5
```

### 2. Evaluate a model | 模型评估

```bash
python evaluate.py --model_dir ../run/bertcrf_best
```

---

## Models Overview | 模型简介

| Model        | Highlights / 特点                                                                             |
| ------------ | ------------------------------------------------------------------------------------------- |
| **BERT-CRF** | Strong baseline for sequence labeling with global decoding.                                 |
| **LEBERT**   | Incorporates external lexicon embeddings to improve recall.                                 |
| **SpanKL**   | Uses span-based classification with KL regularization for better entity boundary detection. |
| **MFME-NER** | Fuses multiple features (lexicon, POS, bigram, type) using memory encoding.                 |
| **FLAT**     | Models character-word lattice structures via flat-lattice transformer layers.               |

---

## Experimental Results | 实验结果

| Model    | Precision | Recall | F1-score |
| -------- | --------- | ------ | -------- |
| BERT-CRF | xx.xx     | xx.xx  | xx.xx    |
| LEBERT   | xx.xx     | xx.xx  | xx.xx    |
| SpanKL   | xx.xx     | xx.xx  | xx.xx    |
| MFME-NER | xx.xx     | xx.xx  | xx.xx    |
| FLAT     | xx.xx     | xx.xx  | xx.xx    |

(*Replace `xx.xx` with actual results from your experiments.*)

---

## Citation | 引用

If you use this repository in your research, please cite the corresponding papers of each model.

---


