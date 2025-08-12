
from typing import List, Dict, Any
import json, os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import jieba

from conll_utils import read_conll

class LEBERTDataset(Dataset):
    """
    Build BERT inputs + lexicon ids (per-char) aligned to token positions.
    - char_lex_ids: [num_chars, K] (top-K lexicon ids per char; 0=pad)
    - token2char_index: [seq_len] mapping token positions to char index (-1 for [CLS]/[SEP]/subword)
    """
    def __init__(self, data_path: str, split: str, pretrained: str, label2id: Dict[str,int],
                 lexicon_json: str, max_len: int = 256, topk: int = 5):
        self.topk = topk
        sents, labels = read_conll(os.path.join(data_path, f"{split}.txt"))
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained)
        with open(lexicon_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.word2id = meta["word2id"]
        # frequency to prioritize (if ties)
        self.freq = {w: i for i, w in enumerate(sorted(self.word2id.keys()))}
        self.label2id = label2id
        self.max_len = max_len
        self.features = []
        for tokens, tags in zip(sents, labels):
            text = "".join(tokens)
            # 1) match lexicon in search mode
            char_words = [[] for _ in range(len(tokens))]  # per-char word ids
            for w, start, end in jieba.tokenize(text, mode='search'):
                if w in self.word2id:
                    wid = self.word2id[w]
                    # end is exclusive
                    for i in range(start, end):
                        if 0 <= i < len(tokens):
                            char_words[i].append(wid)
            # per-char select topK (by word length descending then id as tiebreaker)
            char_lex_ids = []
            for i, lst in enumerate(char_words):
                if not lst:
                    char_lex_ids.append([0]*self.topk)
                else:
                    # keep unique and maybe sort by wid (proxy for freq/length not stored); keep first K
                    # to favor longer words, a simple heuristic: sort by wid ascending (not perfect). Kept simple.
                    uniq = list(dict.fromkeys(lst))[:self.topk]
                    pad = [0]*(self.topk-len(uniq))
                    char_lex_ids.append(uniq + pad)

            # 2) tokenize with alignment
            enc = self.tokenizer(tokens, is_split_into_words=True,
                                 truncation=True, max_length=max_len,
                                 return_offsets_mapping=False)
            word_ids = enc.word_ids()
            labels_ids, valid_mask, prev = [], [], None
            token2char_index = []
            for wid in word_ids:
                if wid is None:
                    labels_ids.append(self.label2id.get("O", 0))
                    valid_mask.append(0)
                    token2char_index.append(-1)
                else:
                    if wid != prev:
                        labels_ids.append(self.label2id.get(tags[wid], 0))
                        valid_mask.append(1)
                        token2char_index.append(wid)
                        prev = wid
                    else:
                        labels_ids.append(self.label2id.get("O", 0))
                        valid_mask.append(0)
                        token2char_index.append(-1)

            self.features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0]*len(enc["input_ids"])),
                "labels": labels_ids,
                "valid_mask": valid_mask,
                "token2char_index": token2char_index,
                "char_lex_ids": char_lex_ids,  # per-char
                "orig_tokens": tokens,
                "orig_labels": tags
            })

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

class CollateLEBERT:
    def __init__(self, pad_id: int, topk: int = 5):
        self.pad_id = pad_id
        self.topk = topk
    def __call__(self, batch: List[Dict[str,Any]]):
        max_seq = max(len(x["input_ids"]) for x in batch)
        max_chars = max(len(x["char_lex_ids"]) for x in batch)
        out = {k: [] for k in batch[0].keys()}
        for item in batch:
            seq_len = len(item["input_ids"])
            pad_seq = max_seq - seq_len
            out["input_ids"].append(item["input_ids"] + [self.pad_id]*pad_seq)
            out["attention_mask"].append(item["attention_mask"] + [0]*pad_seq)
            out["token_type_ids"].append(item["token_type_ids"] + [0]*pad_seq)
            out["labels"].append(item["labels"] + [0]*pad_seq)
            out["valid_mask"].append(item["valid_mask"] + [0]*pad_seq)
            # token2char_index pad with -1
            out["token2char_index"].append(item["token2char_index"] + [-1]*pad_seq)
            # char_lex_ids pad on char dimension
            char_pad = [[0]*self.topk for _ in range(max_chars - len(item["char_lex_ids"]))]
            out["char_lex_ids"].append(item["char_lex_ids"] + char_pad)
            out["orig_tokens"].append(item["orig_tokens"])
            out["orig_labels"].append(item["orig_labels"])
        # convert to tensors
        for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]:
            out[k] = torch.tensor(out[k], dtype=torch.long)
        out["token2char_index"] = torch.tensor(out["token2char_index"], dtype=torch.long)
        out["char_lex_ids"] = torch.tensor(out["char_lex_ids"], dtype=torch.long)
        return out
