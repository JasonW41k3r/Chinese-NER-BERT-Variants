from typing import List, Tuple, Dict
from dataclasses import dataclass

def read_conll(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sents, labels = [], []
    cur_s, cur_l = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if cur_s:
                    sents.append(cur_s); labels.append(cur_l)
                    cur_s, cur_l = [], []
            else:
                parts = line.split(" ")
                if len(parts) < 2: continue
                ch, tag = parts[0], parts[1]
                cur_s.append(ch); cur_l.append(tag)
    if cur_s:
        sents.append(cur_s); labels.append(cur_l)
    return sents, labels

def build_label_map(*label_lists: List[List[str]]) -> Dict[str, int]:
    label_set = set()
    for lab in label_lists:
        for seq in lab:
            label_set.update(seq)
    labels = sorted(label_set)
    if "O" in labels:
        labels.remove("O")
    labels = ["O"] + labels
    id2label = {i: lab for i, lab in enumerate(labels)}
    label2id = {lab: i for i, lab in id2label.items()}
    return {"label2id": label2id, "id2label": id2label, "labels": labels}

@dataclass
class NerExample:
    tokens: List[str]
    labels: List[str]
