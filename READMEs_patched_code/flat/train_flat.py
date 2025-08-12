import os, json, math, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from conll_utils import read_conll, build_label_map
from dataset_flat import FLATDataset, CollateFLAT
from model_flat import FlatLatticeNER

def evaluate(model, loader, id2label, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids","attention_mask","token_type_ids",
                      "node_spans","node_types","node_lex_ids","node_mask",
                      "char_token_pos","char_mask","labels_char"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "node_spans","node_types","node_lex_ids","node_mask",
                                                "char_token_pos","char_mask","labels_char"]})
            preds = out["pred_tags"]; mask = out["mask"].cpu().numpy(); labels = batch["labels_char"].cpu().numpy()
            B = labels.shape[0]
            for i in range(B):
                valid_pos = np.where(mask[i]==1)[0]
                true_seq = [id2label[int(labels[i, p])] for p in valid_pos]
                pred_seq = [id2label[int(x)] for x in preds[i]]
                all_true.append(true_seq); all_pred.append(pred_seq)
    metrics = {
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall":    recall_score(all_true, all_pred, zero_division=0),
        "f1":        f1_score(all_true, all_pred, zero_division=0),
    }
    report = classification_report(all_true, all_pred, digits=4)
    return metrics, report

def train(args):
    set_seed(args.seed)
    # 标签
    train_s, train_l = read_conll(os.path.join(args.data_dir, "train.txt"))
    dev_s, dev_l = read_conll(os.path.join(args.data_dir, "dev.txt"))
    test_s, test_l = read_conll(os.path.join(args.data_dir, "test.txt"))
    lm = build_label_map(train_l, dev_l, test_l)
    label2id, id2label = lm["label2id"], {int(i):l for i,l in lm["id2label"].items()}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # 词表大小
    with open(args.lexicon_json, "r", encoding="utf-8") as f:
        lex_size = json.load(f)["size"]

    tok = BertTokenizerFast.from_pretrained(args.pretrained)
    collate = CollateFLAT(pad_id=tok.pad_token_id)
    train_ds = FLATDataset(args.data_dir, "train", args.pretrained, label2id, args.lexicon_json,
                           max_len=args.max_len, max_words_per_sent=args.max_words)
    dev_ds   = FLATDataset(args.data_dir, "dev", args.pretrained, label2id, args.lexicon_json,
                           max_len=args.max_len, max_words_per_sent=args.max_words)
    test_ds  = FLATDataset(args.data_dir, "test", args.pretrained, label2id, args.lexicon_json,
                           max_len=args.max_len, max_words_per_sent=args.max_words)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # config（记录自定义参数，保存便于复现）
    config = BertConfig.from_pretrained(args.pretrained, num_labels=len(label2id))
    config.lexicon_size = int(lex_size)
    config.lex_dim = int(args.lex_dim)
    config.hidden_dropout_prob = args.dropout

    # 模型
    model = FlatLatticeNER.from_pretrained(
        args.pretrained, config=config,
        num_labels=len(label2id), lexicon_size=lex_size, lex_dim=args.lex_dim,
        n_layers=args.layers, n_heads=args.heads, d_ff=args.ff, max_dist=args.max_dist
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器/调度器
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    t_total = max(1, math.ceil(len(train_loader) * args.epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup), t_total)

    # 训练
    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            for k in ["input_ids","attention_mask","token_type_ids",
                      "node_spans","node_types","node_lex_ids","node_mask",
                      "char_token_pos","char_mask","labels_char"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "node_spans","node_types","node_lex_ids","node_mask",
                                                "char_token_pos","char_mask","labels_char"]})
            loss = out["loss"]
            if (loss is None) or (not torch.isfinite(loss)):
                print(f"[WARN] Non-finite loss at epoch={epoch} step={step}. Skip.")
                optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step()

            # 夹住 CRF 转移，防爆
            with torch.no_grad():
                model.crf.transitions.clamp_(-5.0, 5.0)
                model.crf.start_transitions.clamp_(-5.0, 5.0)
                model.crf.end_transitions.clamp_(-5.0, 5.0)

            optimizer.zero_grad()
            running += float(loss.item())
            if step % args.log_steps == 0:
                print(f"Epoch {epoch} Step {step} | loss={running/args.log_steps:.4f}")
                running = 0.0

        dev_metrics, dev_report = evaluate(model, dev_loader, {i:l for i,l in enumerate(model.config.id2label.values())} if hasattr(model.config,'id2label') else {i:str(i) for i in range(config.num_labels)}, device)
        print(f"[DEV] epoch={epoch} f1={dev_metrics['f1']:.4f} p={dev_metrics['precision']:.4f} r={dev_metrics['recall']:.4f}")
        with open(os.path.join(args.out_dir, f"dev_report_epoch{epoch}.txt"), "w", encoding="utf-8") as f:
            f.write(dev_report)
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_path = os.path.join(args.out_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            tok.save_pretrained(best_path)
            with open(os.path.join(best_path, "dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, ensure_ascii=False, indent=2)

    print("Evaluating best checkpoint on TEST...")
    best_model = FlatLatticeNER.from_pretrained(best_path).to(device)
    test_metrics, test_report = evaluate(best_model, test_loader, {i:l for i,l in enumerate(model.config.id2label.values())} if hasattr(model.config,'id2label') else {i:str(i) for i in range(config.num_labels)}, device)
    print(test_report)
    with open(os.path.join(args.out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./run")
    ap.add_argument("--pretrained", type=str, default="bert-base-chinese")
    ap.add_argument("--lexicon_json", type=str, default="./feats/lexicon.json")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_words", type=int, default=50)  # 每句词节点上限，稳定/省显存
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)  # 稳妥起步
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--lex_dim", type=int, default=100)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_dist", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
