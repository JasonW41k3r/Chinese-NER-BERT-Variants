
"""
Build a simple lexicon from train.txt using jieba (search mode).
- Keep words with length >= 2
- Keep words with frequency >= min_freq
Saves: lexicon.json
"""
import os, json, argparse, collections
import jieba
from conll_utils import read_conll

def main(args):
    train_path = os.path.join(args.data_dir, "train.txt")
    sents, labels = read_conll(train_path)
    # build corpus strings
    corpus = ["".join(s) for s in sents]
    freq = collections.Counter()
    for s in corpus:
        # search mode yields overlapping words
        for w, start, end in jieba.tokenize(s, mode='search'):
            if len(w) >= args.min_word_len:
                freq[w] += 1
    # filter by min_freq
    items = [(w, c) for w, c in freq.items() if c >= args.min_freq]
    # sort by (-freq, -len, word)
    items.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    # id starts from 1; 0 is padding
    word2id = {w: i+1 for i, (w, _) in enumerate(items)}
    id2word = {i: w for w, i in word2id.items()}
    meta = {
        "min_freq": args.min_freq,
        "min_word_len": args.min_word_len,
        "size": len(word2id),
        "word2id": word2id,
        "id2word": id2word
    }
    out_path = os.path.join(args.out_dir, "lexicon.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Lexicon size: {len(word2id)} saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/data")
    parser.add_argument("--out_dir", type=str, default="/mnt/data/lebert")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--min_word_len", type=int, default=2)
    args = parser.parse_args()
    main(args)
