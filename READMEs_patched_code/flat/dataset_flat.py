from typing import List, Dict, Any, Tuple
import os, json
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import torch
from conll_utils import read_conll

def find_all_occurrences(text: str, word: str) -> List[Tuple[int,int]]:
    res = []
    start = 0
    while True:
        idx = text.find(word, start)
        if idx == -1: break
        res.append((idx, idx+len(word)-1))  # inclusive
        start = idx + 1
    return res

class FLATDataset(Dataset):
    """
    为每句构造：
    - BERT输入（按“字”为词，首子词对齐）
    - 扁平词格节点：先放 C 个字节点 (i,i, type=0)，再接若干词节点 (st,ed,type=1,lex_id)
    - char_token_pos: [C] 每个字的首子词 token 下标（-1 表 padding）
    - labels_char: [C] 字级标签 id
    - node_spans: [N,2] 每个节点的 (st,ed) 字下标
    - node_types: [N] 0/1 = char/word
    - node_lex_ids: [N] 词节点的词表 id（字节点为 0）
    """
    def __init__(self, data_dir: str, split: str, pretrained: str, label2id: Dict[str,int],
                 lexicon_json: str, max_len: int = 160, max_words_per_sent: int = 50):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained)
        self.label2id = label2id
        self.max_len = max_len
        self.max_words_per_sent = max_words_per_sent

        with open(lexicon_json, "r", encoding="utf-8") as f:
            lx = json.load(f)
        self.lex_word2id = lx["word2id"]
        self.lex_size = lx["size"]

        sents, labels = read_conll(os.path.join(data_dir, f"{split}.txt"))
        self.features = []
        for tokens, tags in zip(sents, labels):
            text = "".join(tokens)
            C = len(tokens)

            # BERT 编码（按字对齐）
            enc = self.tokenizer(tokens, is_split_into_words=True,
                                 truncation=True, max_length=max_len,
                                 return_offsets_mapping=False)
            word_ids = enc.word_ids()
            char_token_pos = [-1]*C
            prev = None
            for ti, wid in enumerate(word_ids):
                if wid is None: continue
                if wid != prev:
                    if 0 <= wid < C and char_token_pos[wid] == -1:
                        char_token_pos[wid] = ti
                    prev = wid

            # 构建词节点（扫描 lexicon），按长度优先限制数量
            word_spans = []
            # 优先长词：减少冗余，稳定训练
            for w, wid in sorted(self.lex_word2id.items(), key=lambda x: -len(x[0])):
                for st, ed in find_all_occurrences(text, w):
                    word_spans.append((st, ed, wid))
                if len(word_spans) >= self.max_words_per_sent:
                    break
            if len(word_spans) > self.max_words_per_sent:
                word_spans = word_spans[:self.max_words_per_sent]

            # 扁平节点：先所有字节点，再词节点
            node_spans = [(i, i) for i in range(C)] + [(st, ed) for st,ed,_ in word_spans]
            node_types = [0]*C + [1]*len(word_spans)
            node_lex_ids = [0]*C + [wid for _,_,wid in word_spans]

            labels_char = [self.label2id.get(t, 0) for t in tags]

            self.features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0]*len(enc["input_ids"])),
                "char_token_pos": char_token_pos,
                "labels_char": labels_char,
                "node_spans": node_spans,
                "node_types": node_types,
                "node_lex_ids": node_lex_ids,
                "n_input": len(enc["input_ids"]),
                "n_chars": C,
                "n_nodes": len(node_spans),
            })

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

class CollateFLAT:
    def __init__(self, pad_id:int):
        self.pad_id = pad_id
    def __call__(self, batch: List[Dict[str,Any]]):
        max_seq = max(x["n_input"] for x in batch)
        max_chars = max(x["n_chars"] for x in batch)
        max_nodes = max(x["n_nodes"] for x in batch)
        out = {
            "input_ids": [], "attention_mask": [], "token_type_ids": [],
            "char_token_pos": [], "char_mask": [], "labels_char": [],
            "node_spans": [], "node_types": [], "node_lex_ids": [],
            "node_mask": [], "char_len": [], "node_len": []
        }
        for item in batch:
            # BERT 序列 pad
            pad_seq = max_seq - item["n_input"]
            out["input_ids"].append(item["input_ids"] + [self.pad_id]*pad_seq)
            out["attention_mask"].append(item["attention_mask"] + [0]*pad_seq)
            out["token_type_ids"].append(item["token_type_ids"] + [0]*pad_seq)

            # 字级 pad
            pad_chars = max_chars - item["n_chars"]
            out["char_token_pos"].append(item["char_token_pos"] + [-1]*pad_chars)
            out["labels_char"].append(item["labels_char"] + [0]*pad_chars)
            out["char_mask"].append([1]*item["n_chars"] + [0]*pad_chars)
            out["char_len"].append(item["n_chars"])

            # 节点 pad
            pad_nodes = max_nodes - item["n_nodes"]
            ns = item["node_spans"] + [[0,0]]*pad_nodes
            out["node_spans"].append(ns)
            out["node_types"].append(item["node_types"] + [0]*pad_nodes)
            out["node_lex_ids"].append(item["node_lex_ids"] + [0]*pad_nodes)
            out["node_mask"].append([1]*item["n_nodes"] + [0]*pad_nodes)
            out["node_len"].append(item["n_nodes"])

        # to tensor
        t = torch.tensor
        out["input_ids"] = t(out["input_ids"], dtype=torch.long)
        out["attention_mask"] = t(out["attention_mask"], dtype=torch.long)
        out["token_type_ids"] = t(out["token_type_ids"], dtype=torch.long)
        out["char_token_pos"] = t(out["char_token_pos"], dtype=torch.long)
        out["labels_char"] = t(out["labels_char"], dtype=torch.long)
        out["char_mask"] = t(out["char_mask"], dtype=torch.long)
        out["node_spans"] = t(out["node_spans"], dtype=torch.long)
        out["node_types"] = t(out["node_types"], dtype=torch.long)
        out["node_lex_ids"] = t(out["node_lex_ids"], dtype=torch.long)
        out["node_mask"] = t(out["node_mask"], dtype=torch.long)
        out["char_len"] = t(out["char_len"], dtype=torch.long)
        out["node_len"] = t(out["node_len"], dtype=torch.long)
        return out
