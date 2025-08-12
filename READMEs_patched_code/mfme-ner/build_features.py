"""
从 train.txt 自动构建：
- lexicon.json: jieba 搜索模式抽取的词（len>=2, min_freq 次数以上）
- bigram.json: 字符 bigram 词表（含 <EOS>）
- pos_vocab.json: jieba.posseg 的词性集合
"""
import os, json, argparse, collections, jieba
import jieba.posseg as pseg
from conll_utils import read_conll

def build_lexicon(corpus, min_word_len=2, min_freq=2):
    freq = collections.Counter()
    for s in corpus:
        for w, st, ed in jieba.tokenize(s, mode='search'):
            if len(w) >= min_word_len:
                freq[w] += 1
    items = [(w,c) for w,c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    word2id = {w:i+1 for i,(w,_) in enumerate(items)}  # 0=PAD
    return {"size": len(word2id), "word2id": word2id}

def build_bigrams(corpus):
    freq = collections.Counter()
    for s in corpus:
        for i,ch in enumerate(s):
            nxt = s[i+1] if i+1 < len(s) else "<EOS>"
            bg = ch + nxt
            freq[bg] += 1
    items = sorted(list(freq.keys()))
    bigram2id = {bg:i+1 for i,bg in enumerate(items)}  # 0=PAD
    return {"size": len(bigram2id), "bigram2id": bigram2id}

def build_pos_vocab(corpus):
    tags = set()
    for s in corpus:
        for w, t in pseg.cut(s):
            tags.add(t)
    pos2id = {t:i+1 for i,t in enumerate(sorted(tags))}  # 0=PAD
    return {"size": len(pos2id), "pos2id": pos2id}

def main(args):
    train_path = os.path.join(args.data_dir, "train.txt")
    sents, _ = read_conll(train_path)
    corpus = ["".join(s) for s in sents]

    lex = build_lexicon(corpus, args.min_word_len, args.min_freq)
    bi  = build_bigrams(corpus)
    pos = build_pos_vocab(corpus)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "lexicon.json"), "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "bigram.json"), "w", encoding="utf-8") as f:
        json.dump(bi, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "pos_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(pos, f, ensure_ascii=False, indent=2)
    print("Saved lexicon/bigram/pos to:", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./feats")
    ap.add_argument("--min_word_len", type=int, default=2)
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()
    main(args)
