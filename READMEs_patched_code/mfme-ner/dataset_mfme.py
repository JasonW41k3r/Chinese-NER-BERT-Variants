from typing import List, Dict, Any
import os, json, string
import jieba
import jieba.posseg as pseg
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from conll_utils import read_conll

def char_type_id(ch: str) -> int:
    if ch.isspace(): return 1
    if '\u4e00' <= ch <= '\u9fff': return 2  # CJK
    if ch.isdigit(): return 3
    if ch.isalpha(): return 4
    if ch in string.punctuation: return 5
    return 6

class MFMEDataset(Dataset):
    """
    生成：
    - BERT 对齐 (只在首子词打标签)
    - char_lex_ids: [C, K] 每个字符覆盖它的前K个词ID（来自 lexicon）
    - char_bigram_ids: [C] 当前字符与下一个字符 bigram 的ID（最后一个与 <EOS>）
    - char_pos_ids: [C] 该字符所在分词的 POS id（jieba.posseg）
    - char_type_ids: [C] 字符类型 id
    - token2char_index: [L] token->char 映射（首子词）其余为 -1
    """
    def __init__(self, data_dir: str, split: str, pretrained: str, label2id: Dict[str,int],
                 lexicon_json: str, bigram_json: str, pos_json: str,
                 max_len: int = 256, topk: int = 5):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained)
        self.label2id = label2id
        self.max_len = max_len
        self.topk = topk

        # load features
        with open(lexicon_json, "r", encoding="utf-8") as f:
            self.lex = json.load(f)["word2id"]
        with open(bigram_json, "r", encoding="utf-8") as f:
            self.bigram2id = json.load(f)["bigram2id"]
        with open(pos_json, "r", encoding="utf-8") as f:
            self.pos2id = json.load(f)["pos2id"]

        sents, labels = read_conll(os.path.join(data_dir, f"{split}.txt"))
        self.features = []
        for tokens, tags in zip(sents, labels):
            text = "".join(tokens)
            n = len(tokens)

            # char_lex_ids
            per_char_lex = [[] for _ in range(n)]
            for w, st, ed in jieba.tokenize(text, mode='search'):
                wid = self.lex.get(w, 0)
                if wid == 0: continue
                for i in range(st, ed):
                    if 0 <= i < n:
                        per_char_lex[i].append(wid)
            char_lex_ids = []
            for lst in per_char_lex:
                # 去重，截断到 topk
                seen = []
                for x in lst:
                    if x not in seen:
                        seen.append(x)
                    if len(seen) >= self.topk: break
                pad = [0]*(self.topk-len(seen))
                char_lex_ids.append(seen + pad)

            # bigram ids
            char_bigram_ids = []
            for i,ch in enumerate(tokens):
                nxt = tokens[i+1] if i+1 < n else "<EOS>"
                bg = ch + nxt
                char_bigram_ids.append(self.bigram2id.get(bg, 0))

            # pos ids（按分词覆盖到字符）
            char_pos_ids = [0]*n
            offset = 0
            for w, t in pseg.cut(text):
                for i in range(len(w)):
                    if offset + i < n:
                        char_pos_ids[offset+i] = self.pos2id.get(t, 0)
                offset += len(w)

            # char type
            char_type_ids = [char_type_id(ch) for ch in tokens]

            # tokenize & align
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
                "char_lex_ids": char_lex_ids,
                "char_bigram_ids": char_bigram_ids,
                "char_pos_ids": char_pos_ids,
                "char_type_ids": char_type_ids,
                "orig_tokens": tokens,
                "orig_labels": tags
            })

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

class CollateMFME:
    def __init__(self, pad_id:int, topk:int=5):
        self.pad_id = pad_id
        self.topk = topk
    def __call__(self, batch: List[Dict[str,Any]]):
        max_seq = max(len(x["input_ids"]) for x in batch)
        max_chars = max(len(x["char_lex_ids"]) for x in batch)
        out = {k: [] for k in batch[0].keys()}
        for item in batch:
            seq_len = len(item["input_ids"])
            pad_seq = max_seq - seq_len
            for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]:
                if k in ["attention_mask","labels","valid_mask"]:
                    pad_val = 0
                elif k == "token_type_ids":
                    pad_val = 0
                else:
                    pad_val = self.pad_id
                out[k].append(item[k] + [pad_val]*pad_seq)
            out["token2char_index"].append(item["token2char_index"] + [-1]*pad_seq)

            # char-level paddings
            char_pad_rows = max_chars - len(item["char_lex_ids"])
            out["char_lex_ids"].append(item["char_lex_ids"] + [[0]*self.topk for _ in range(char_pad_rows)])
            out["char_bigram_ids"].append(item["char_bigram_ids"] + [0]*char_pad_rows)
            out["char_pos_ids"].append(item["char_pos_ids"] + [0]*char_pad_rows)
            out["char_type_ids"].append(item["char_type_ids"] + [0]*char_pad_rows)

            out["orig_tokens"].append(item["orig_tokens"])
            out["orig_labels"].append(item["orig_labels"])

        # to tensor
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        out["token_type_ids"] = torch.tensor(out["token_type_ids"], dtype=torch.long)
        out["labels"] = torch.tensor(out["labels"], dtype=torch.long)
        out["valid_mask"] = torch.tensor(out["valid_mask"], dtype=torch.long)
        out["token2char_index"] = torch.tensor(out["token2char_index"], dtype=torch.long)
        out["char_lex_ids"] = torch.tensor(out["char_lex_ids"], dtype=torch.long)
        out["char_bigram_ids"] = torch.tensor(out["char_bigram_ids"], dtype=torch.long)
        out["char_pos_ids"] = torch.tensor(out["char_pos_ids"], dtype=torch.long)
        out["char_type_ids"] = torch.tensor(out["char_type_ids"], dtype=torch.long)
        return out
