from typing import List, Dict, Any
import random, os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from conll_utils import read_conll
from spans import tags_to_spans

class SpanKLDataset(Dataset):
    def __init__(self, data_dir: str, split: str, pretrained: str, label2id: Dict[str,int],
                 max_len: int = 256, max_span_len: int = 8, neg_ratio: int = 3, max_negs:int=100):
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained)
        self.label2id = label2id
        self.max_len = max_len
        self.max_span_len = max_span_len
        self.neg_ratio = neg_ratio
        self.max_negs = max_negs

        sents, labels = read_conll(os.path.join(data_dir, f"{split}.txt"))
        self.features = []
        for tokens, tags in zip(sents, labels):
            enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                                 max_length=max_len, return_offsets_mapping=False)
            word_ids = enc.word_ids()
            valid_positions = []
            tok2char_index = [-1]*len(enc["input_ids"])
            prev = None
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev:
                    valid_positions.append(i)
                    tok2char_index[i] = wid
                    prev = wid

            n_chars = len(tokens)
            gold_char_spans = tags_to_spans(tags)

            # char -> token 位置映射（首子词位置）
            char2tok = {k: valid_positions[k] for k in range(min(len(valid_positions), n_chars))}

            gold_tok_spans = []
            for st, ed, lab in gold_char_spans:
                if st in char2tok and ed in char2tok:
                    gold_tok_spans.append((char2tok[st], char2tok[ed], lab))

            # 枚举候选 span（基于 char，再映射到 token）
            cand_tok_spans = []
            for st in range(n_chars):
                max_ed = min(n_chars-1, st + self.max_span_len - 1)
                for ed in range(st, max_ed+1):
                    if st in char2tok and ed in char2tok:
                        cand_tok_spans.append((char2tok[st], char2tok[ed]))

            # 标注正负：与 gold 完全重合的记为其类，否则 'O'
            pos, neg = [], []
            for (ts, te) in cand_tok_spans:
                lab = "O"
                for (gts, gte, glab) in gold_tok_spans:
                    if ts == gts and te == gte:
                        lab = glab
                        break
                lab_id = self.label2id.get(lab, 0)
                if lab != "O":
                    pos.append(((ts, te), lab_id))
                else:
                    neg.append(((ts, te), 0))

            # 负采样
            if len(pos) == 0:
                sampled_neg = random.sample(neg, min(self.max_negs, len(neg))) if len(neg)>0 else []
            else:
                sampled_neg = random.sample(neg, min(len(pos)*self.neg_ratio, len(neg)))
            spans = pos + sampled_neg
            random.shuffle(spans)
            if len(spans) == 0 and len(valid_positions) > 0:
                ts = te = valid_positions[0]
                spans = [((ts, te), 0)]

            span_positions = [list(p) for (p, _) in spans]
            span_labels = [lb for (_, lb) in spans]

            self.features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0]*len(enc["input_ids"])),
                "tok2char_index": tok2char_index,
                "span_positions": span_positions,
                "span_labels": span_labels,
                "n_input": len(enc["input_ids"]),
                "n_chars": n_chars,
                "orig_labels": tags
            })

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

class CollateSpan:
    def __init__(self, pad_id:int):
        self.pad_id = pad_id
    def __call__(self, batch: List[Dict[str,Any]]):
        max_len = max(x["n_input"] for x in batch)
        out = {
            "input_ids": [], "attention_mask": [], "token_type_ids": [],
            "tok2char_index": [], "span_positions": [], "span_labels": [],
            "span_batch_idx": [], "n_chars": [], "orig_labels": []
        }
        for bi, item in enumerate(batch):
            pad = max_len - item["n_input"]
            out["input_ids"].append(item["input_ids"] + [self.pad_id]*pad)
            out["attention_mask"].append(item["attention_mask"] + [0]*pad)
            out["token_type_ids"].append(item["token_type_ids"] + [0]*pad)
            out["tok2char_index"].append(item["tok2char_index"] + [-1]*pad)
            # 展平 spans
            for idx, (s,e) in enumerate(item["span_positions"]):
                out["span_positions"].append([s,e])
                out["span_labels"].append(item["span_labels"][idx])
                out["span_batch_idx"].append(bi)
            out["n_chars"].append(item["n_chars"])
            out["orig_labels"].append(item["orig_labels"])

        import torch
        out["input_ids"] = torch.tensor(out["input_ids"], dtype=torch.long)
        out["attention_mask"] = torch.tensor(out["attention_mask"], dtype=torch.long)
        out["token_type_ids"] = torch.tensor(out["token_type_ids"], dtype=torch.long)
        out["tok2char_index"] = torch.tensor(out["tok2char_index"], dtype=torch.long)

        if len(out["span_positions"]) == 0:
            out["span_positions"] = torch.zeros((0,2), dtype=torch.long)
            out["span_labels"] = torch.zeros((0,), dtype=torch.long)
            out["span_batch_idx"] = torch.zeros((0,), dtype=torch.long)
        else:
            out["span_positions"] = torch.tensor(out["span_positions"], dtype=torch.long)
            out["span_labels"] = torch.tensor(out["span_labels"], dtype=torch.long)
            out["span_batch_idx"] = torch.tensor(out["span_batch_idx"], dtype=torch.long)
        return out
