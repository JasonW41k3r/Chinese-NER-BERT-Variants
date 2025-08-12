"""
从 train.txt 自动构建 lexicon（搜索模式分词，过滤太短/低频）。
输出: feats/lexicon.json: {"size": int, "word2id": {...}, "freq": {...}}
"""
import os, json, argparse, collections, jieba
from conll_utils import read_conll

def main(args):
    sents, _ = read_conll(os.path.join(args.data_dir, "train.txt"))
    corpus = ["".join(x) for x in sents]
    cnt = collections.Counter()
    for text in corpus:
        for w, st, ed in jieba.tokenize(text, mode='search'):
            if len(w) >= args.min_word_len:
                cnt[w] += 1
    kept = [(w,c) for w,c in cnt.items() if c >= args.min_freq]
    kept.sort(key=lambda x:(-x[1], -len(x[0]), x[0]))
    if args.max_words > 0:
        kept = kept[:args.max_words]
    word2id = {w:i+1 for i,(w,_) in enumerate(kept)}  # 0 = PAD
    freq = {w:c for w,c in kept}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "lexicon.json"), "w", encoding="utf-8") as f:
        json.dump({"size": len(word2id), "word2id": word2id, "freq": freq}, f, ensure_ascii=False, indent=2)
    print(f"Saved lexicon: {len(word2id)} words -> {os.path.join(args.out_dir, 'lexicon.json')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./feats")
    ap.add_argument("--min_word_len", type=int, default=2)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_words", type=int, default=50000)  # 上限，防 OOM
    args = ap.parse_args()
    main(args)
