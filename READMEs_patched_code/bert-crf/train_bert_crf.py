# train_bert_crf.py
import os, json, math, argparse, numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertConfig, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from conll_utils import read_conll, build_label_map
from model_bert_crf import BertCRFForTokenClassification

@dataclass
class Collate:
    pad_id: int
    label_pad_id: int = 0  # 形状对齐，真实忽略靠CRF的mask
    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        out = {k: [] for k in batch[0].keys()}
        for item in batch:
            pad = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [self.pad_id]*pad)
            out["attention_mask"].append(item["attention_mask"] + [0]*pad)
            out["token_type_ids"].append(item["token_type_ids"] + [0]*pad)
            out["labels"].append(item["labels"] + [self.label_pad_id]*pad)
            out["valid_mask"].append(item["valid_mask"] + [0]*pad)
            out["orig_labels"].append(item["orig_labels"])
            out["orig_tokens"].append(item["orig_tokens"])
        for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]:
            out[k] = torch.tensor(out[k], dtype=torch.long)
        return out

class NERDataset(Dataset):
    def __init__(self, sents, labels, tokenizer, label2id, max_len=256):
        self.features = []
        for tokens, tags in zip(sents, labels):
            enc = tokenizer(tokens, is_split_into_words=True,
                            truncation=True, max_length=max_len,
                            return_offsets_mapping=False)
            word_ids = enc.word_ids()  # 子词->词/字索引
            lab_ids, valid_mask, prev_word = [], [], None
            for wid in word_ids:
                if wid is None:
                    lab_ids.append(0); valid_mask.append(0)  # [CLS]/[SEP]
                else:
                    if wid != prev_word:
                        lab_ids.append(label2id.get(tags[wid], 0))
                        valid_mask.append(1)
                        prev_word = wid
                    else:
                        lab_ids.append(0); valid_mask.append(0)  # 后续子词
            self.features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "token_type_ids": enc.get("token_type_ids", [0]*len(enc["input_ids"])),
                "labels": lab_ids,
                "valid_mask": valid_mask,
                "orig_labels": tags,
                "orig_tokens": tokens
            })
    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

def evaluate(model, loader, id2label, device):
    model.eval()
    all_true, all_pred = [], []
    token_total, token_correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]})
            pred_paths = out["pred_tags"]
            mask = out["mask"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            B = labels.shape[0]
            for i in range(B):
                true_seq = [id2label[int(labels[i, pos])]
                            for pos in np.where(mask[i]==1)[0]]
                pred_seq = [id2label[int(idx)] for idx in pred_paths[i]]
                token_total += len(true_seq)
                token_correct += sum([t==p for t,p in zip(true_seq, pred_seq)])
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    metrics = {
        "precision": precision_score(all_true, all_pred),
        "recall":    recall_score(all_true, all_pred),
        "f1":        f1_score(all_true, all_pred),
        "token_acc": (token_correct / max(1, token_total))
    }
    report_str = classification_report(all_true, all_pred, digits=4)
    return metrics, report_str

def train(args):
    set_seed(args.seed)
    # 1) 读数据
    train_s, train_l = read_conll(os.path.join(args.data_dir, "train.txt"))
    dev_s,   dev_l   = read_conll(os.path.join(args.data_dir, "dev.txt"))
    test_s,  test_l  = read_conll(os.path.join(args.data_dir, "test.txt"))

    lm = build_label_map(train_l, dev_l, test_l)
    label2id, id2label = lm["label2id"], {int(i):l for i,l in lm["id2label"].items()}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # 2) tokenizer 与 Dataset
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained)
    train_ds = NERDataset(train_s, train_l, tokenizer, label2id, max_len=args.max_len)
    dev_ds   = NERDataset(dev_s,   dev_l,   tokenizer, label2id, max_len=args.max_len)
    test_ds  = NERDataset(test_s,  test_l,  tokenizer, label2id, max_len=args.max_len)

    # 3) 模型
    config = BertConfig.from_pretrained(args.pretrained,
                                        num_labels=len(label2id),
                                        id2label=id2label, label2id=label2id)
    model = BertCRFForTokenClassification.from_pretrained(args.pretrained, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) 优化器/调度器/Loader
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)

    collate = Collate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    t_total   = math.ceil(len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup), t_total)

    # 5) 训练与验证
    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask"]})
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            if step % args.log_steps == 0:
                print(f"Epoch {epoch} Step {step} | loss={total_loss/args.log_steps:.4f}")
                total_loss = 0.0

        # 每轮在dev评估
        dev_metrics, dev_report = evaluate(model, dev_loader, id2label, device)
        print(f"[DEV] epoch={epoch} f1={dev_metrics['f1']:.4f} p={dev_metrics['precision']:.4f} r={dev_metrics['recall']:.4f} acc={dev_metrics['token_acc']:.4f}")
        with open(os.path.join(args.out_dir, f"dev_report_epoch{epoch}.txt"), "w", encoding="utf-8") as f:
            f.write(dev_report)

        # 取best（按dev F1）
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_path = os.path.join(args.out_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            with open(os.path.join(best_path, "dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, ensure_ascii=False, indent=2)

    # 6) 用best在test上评测
    print("Evaluating best checkpoint on TEST...")
    best_model = BertCRFForTokenClassification.from_pretrained(best_path).to(device)
    test_metrics, test_report = evaluate(best_model, test_loader, id2label, device)
    print(test_report)
    with open(os.path.join(args.out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs/bert_crf/run1")
    parser.add_argument("--pretrained", type=str, default="bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
