
# LEBERT (Lightweight) — Lexicon-Enhanced BERT + CRF

This is a compact, **no-external-dictionary** LEBERT-style pipeline:
- Builds a lexicon from `train.txt` with jieba (search mode)
- Trains **BERT + lexicon embedding fusion + CRF**
- Evaluates with **seqeval** (micro P/R/F1 + per-class report)

## Install
```bash
conda create -n lebert python=3.10 -y
conda activate lebert
# choose your CUDA wheel accordingly
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install transformers pytorch-crf seqeval jieba accelerate
```

## Data
Place `train.txt`, `dev.txt`, `test.txt` in your data dir (default `/mnt/data`).
Each line: `<char> <label>` (space separated), sentences split by blank lines.

## Build Lexicon
```bash
python build_lexicon.py --data_dir /mnt/data --out_dir /mnt/data/lebert --min_freq 2 --min_word_len 2
```
This creates `/mnt/data/lebert/lexicon.json`.

## Train & Evaluate
```bash
python train_lebert.py \
  --data_dir /mnt/data \
  --out_dir /mnt/data/lebert/run \
  --pretrained bert-base-chinese \
  --lexicon_json /mnt/data/lebert/lexicon.json \
  --batch_size 8 --eval_batch_size 16 --epochs 5 --max_len 160 --topk 5 --lex_dim 100
```
Tips for small GPUs (e.g., RTX 2050 4GB):
- Use `--batch_size 8`, `--max_len 128~160`
- You can also lower `--lex_dim` to 64

## Outputs
- Best checkpoint: `/mnt/data/lebert/run/best/`
- Dev reports: `dev_report_epoch*.txt`
- Test: `test_report.txt`, `test_metrics.json`
- CSV summary: `report_aggregate.py` → `metrics_summary.csv`
