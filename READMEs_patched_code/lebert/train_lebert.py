
import os, json, math, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, set_seed
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from conll_utils import read_conll, build_label_map
from dataset_lebert import LEBERTDataset, CollateLEBERT
from model_lebert_crf import LeBertCRF

def evaluate(model, loader, id2label, device):
    model.eval()
    all_true, all_pred = [], []
    token_total, token_correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask","token2char_index","char_lex_ids"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "labels","valid_mask","token2char_index","char_lex_ids"]})
            pred_paths = out["pred_tags"]
            mask = out["mask"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            B = labels.shape[0]
            for i in range(B):
                true_seq = [id2label[int(labels[i, pos])] for pos in np.where(mask[i]==1)[0]]
                pred_seq = [id2label[int(idx)] for idx in pred_paths[i]]
                token_total += len(true_seq)
                token_correct += sum([t==p for t,p in zip(true_seq, pred_seq)])
                all_true.append(true_seq)
                all_pred.append(pred_seq)
    metrics = {
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall":    recall_score(all_true, all_pred, zero_division=0),
        "f1":        f1_score(all_true, all_pred, zero_division=0),
        "token_acc": (token_correct / max(1, token_total))
    }
    report_str = classification_report(all_true, all_pred, digits=4)
    return metrics, report_str

def train(args):
    set_seed(args.seed)
    # 0) label map
    train_s, train_l = read_conll(os.path.join(args.data_dir, "train.txt"))
    dev_s, dev_l = read_conll(os.path.join(args.data_dir, "dev.txt"))
    test_s, test_l = read_conll(os.path.join(args.data_dir, "test.txt"))
    lm = build_label_map(train_l, dev_l, test_l)
    label2id, id2label = lm["label2id"], {int(i):l for i,l in lm["id2label"].items()}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # 1) datasets (ensure lexicon exists)
    if not os.path.exists(args.lexicon_json):
        raise FileNotFoundError(f"Lexicon not found: {args.lexicon_json}. Run build_lexicon.py first.")
    # read lexicon size
    with open(args.lexicon_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lex_size = meta["size"]

    train_ds = LEBERTDataset(args.data_dir, "train", args.pretrained, label2id, args.lexicon_json,
                             max_len=args.max_len, topk=args.topk)
    dev_ds   = LEBERTDataset(args.data_dir, "dev", args.pretrained, label2id, args.lexicon_json,
                             max_len=args.max_len, topk=args.topk)
    test_ds  = LEBERTDataset(args.data_dir, "test", args.pretrained, label2id, args.lexicon_json,
                             max_len=args.max_len, topk=args.topk)

    collate = CollateLEBERT(pad_id=0, topk=args.topk)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # 2) model
    config = BertConfig.from_pretrained(args.pretrained, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    model = LeBertCRF.from_pretrained(args.pretrained, config=config, lexicon_size=lex_size, lex_dim=args.lex_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) optimizer/scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    t_total = math.ceil(len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup), t_total)

    # 4) train
    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            for k in ["input_ids","attention_mask","token_type_ids","labels","valid_mask","token2char_index","char_lex_ids"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "labels","valid_mask","token2char_index","char_lex_ids"]})
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            if step % args.log_steps == 0:
                print(f"Epoch {epoch} Step {step} | loss={total_loss/args.log_steps:.4f}")
                total_loss = 0.0

        dev_metrics, dev_report = evaluate(model, dev_loader, id2label, device)
        print(f"[DEV] epoch={epoch} f1={dev_metrics['f1']:.4f} p={dev_metrics['precision']:.4f} r={dev_metrics['recall']:.4f} acc={dev_metrics['token_acc']:.4f}")
        with open(os.path.join(args.out_dir, f"dev_report_epoch{epoch}.txt"), "w", encoding="utf-8") as f:
            f.write(dev_report)
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_path = os.path.join(args.out_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            with open(os.path.join(best_path, "dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, ensure_ascii=False, indent=2)

    # 5) test
    print("Evaluating best checkpoint on TEST...")
    with open(args.lexicon_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lex_size = meta["size"]
    best_model = LeBertCRF.from_pretrained(best_path, lexicon_size=lex_size, lex_dim=args.lex_dim).to(device)
    test_metrics, test_report = evaluate(best_model, test_loader, id2label, device)
    print(test_report)
    with open(os.path.join(args.out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/data")
    parser.add_argument("--out_dir", type=str, default="/mnt/data/lebert/run")
    parser.add_argument("--pretrained", type=str, default="bert-base-chinese")
    parser.add_argument("--lexicon_json", type=str, default="/mnt/data/lebert/lexicon.json")
    parser.add_argument("--lex_dim", type=int, default=100)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
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
