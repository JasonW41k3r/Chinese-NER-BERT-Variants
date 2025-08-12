# train_mfme.py
import os, json, math, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from conll_utils import read_conll, build_label_map
from dataset_mfme import MFMEDataset, CollateMFME
from model_mfme_crf import MFMEForTokenClassification

def evaluate(model, loader, id2label, device):
    model.eval()
    all_true, all_pred = [], []
    tok_total, tok_correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids","attention_mask","token_type_ids",
                      "labels","valid_mask","token2char_index",
                      "char_lex_ids","char_bigram_ids","char_pos_ids","char_type_ids"]:
                batch[k] = batch[k].to(device)
            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "labels","valid_mask","token2char_index",
                                                "char_lex_ids","char_bigram_ids","char_pos_ids","char_type_ids"]})
            pred_paths = out["pred_tags"]
            mask = out["mask"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            B = labels.shape[0]
            for i in range(B):
                true_seq = [id2label[int(labels[i, pos])] for pos in np.where(mask[i]==1)[0]]
                pred_seq = [id2label[int(idx)] for idx in pred_paths[i]]
                tok_total += len(true_seq)
                tok_correct += sum([t==p for t,p in zip(true_seq, pred_seq)])
                all_true.append(true_seq)
                all_pred.append(pred_seq)
    metrics = {
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall":    recall_score(all_true, all_pred, zero_division=0),
        "f1":        f1_score(all_true, all_pred, zero_division=0),
        "token_acc": (tok_correct / max(1, tok_total))
    }
    report = classification_report(all_true, all_pred, digits=4)
    return metrics, report

def train(args):
    set_seed(args.seed)
    # 1) 标签映射
    train_s, train_l = read_conll(os.path.join(args.data_dir, "train.txt"))
    dev_s, dev_l = read_conll(os.path.join(args.data_dir, "dev.txt"))
    test_s, test_l = read_conll(os.path.join(args.data_dir, "test.txt"))
    lm = build_label_map(train_l, dev_l, test_l)
    label2id, id2label = lm["label2id"], {int(i):l for i,l in lm["id2label"].items()}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # 2) 特征词表大小
    with open(args.lexicon_json, "r", encoding="utf-8") as f:
        lex_size = json.load(f)["size"]
    with open(args.bigram_json, "r", encoding="utf-8") as f:
        bigram_size = json.load(f)["size"]
    with open(args.pos_json, "r", encoding="utf-8") as f:
        pos_size = json.load(f)["size"]
    type_size = 6  # 我们定义的字符类型种类数

    # 3) Datasets / DataLoaders
    tok = BertTokenizerFast.from_pretrained(args.pretrained)
    collate = CollateMFME(pad_id=tok.pad_token_id, topk=args.topk)

    train_ds = MFMEDataset(args.data_dir, "train", args.pretrained, label2id,
                           args.lexicon_json, args.bigram_json, args.pos_json,
                           max_len=args.max_len, topk=args.topk)
    dev_ds   = MFMEDataset(args.data_dir, "dev", args.pretrained, label2id,
                           args.lexicon_json, args.bigram_json, args.pos_json,
                           max_len=args.max_len, topk=args.topk)
    test_ds  = MFMEDataset(args.data_dir, "test", args.pretrained, label2id,
                           args.lexicon_json, args.bigram_json, args.pos_json,
                           max_len=args.max_len, topk=args.topk)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # 4) Config（写入自定义超参，便于追踪；但当前模型构造仍需显式传参）
    config = BertConfig.from_pretrained(args.pretrained, num_labels=len(label2id),
                                        id2label=id2label, label2id=label2id)
    config.lexicon_size = int(lex_size)
    config.bigram_size  = int(bigram_size)
    config.pos_size     = int(pos_size)
    config.type_size    = int(type_size)
    config.lex_dim      = int(args.lex_dim)
    config.bigram_dim   = int(args.bigram_dim)
    config.pos_dim      = int(args.pos_dim)
    config.type_dim     = int(args.type_dim)
    config.mem_hidden   = int(args.mem_hidden)

    # 5) Model（训练阶段显式传参）
    model = MFMEForTokenClassification.from_pretrained(
        args.pretrained, config=config,
        num_labels=len(label2id),
        lexicon_size=lex_size, bigram_size=bigram_size, pos_size=pos_size, type_size=type_size,
        lex_dim=args.lex_dim, bigram_dim=args.bigram_dim, pos_dim=args.pos_dim, type_dim=args.type_dim,
        mem_hidden=args.mem_hidden
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 6) Optim/Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    t_total = max(1, math.ceil(len(train_loader) * args.epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup), t_total)

    # 7) Train
    best_f1, best_path = -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            for k in ["input_ids","attention_mask","token_type_ids",
                      "labels","valid_mask","token2char_index",
                      "char_lex_ids","char_bigram_ids","char_pos_ids","char_type_ids"]:
                batch[k] = batch[k].to(device)

            out = model(**{k:batch[k] for k in ["input_ids","attention_mask","token_type_ids",
                                                "labels","valid_mask","token2char_index",
                                                "char_lex_ids","char_bigram_ids","char_pos_ids","char_type_ids"]})
            loss = out["loss"]

            # Non-finite 哨兵
            if (loss is None) or (not torch.isfinite(loss)):
                print(f"[WARN] Non-finite loss at epoch={epoch} step={step}. Skip this step.")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # —— 夹住 CRF 转移参数，防止爆炸 —— #
            with torch.no_grad():
                model.crf.transitions.clamp_(-5.0, 5.0)
                model.crf.start_transitions.clamp_(-5.0, 5.0)
                model.crf.end_transitions.clamp_(-5.0, 5.0)

            optimizer.zero_grad()
            total_loss += float(loss.item())
            if step % args.log_steps == 0:
                print(f"Epoch {epoch} Step {step} | loss={total_loss/args.log_steps:.4f}")
                total_loss = 0.0

        # DEV
        dev_metrics, dev_report = evaluate(model, dev_loader, id2label, device)
        print(f"[DEV] epoch={epoch} f1={dev_metrics['f1']:.4f} p={dev_metrics['precision']:.4f} r={dev_metrics['recall']:.4f} acc={dev_metrics['token_acc']:.4f}")
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

    # 8) Test（显式传入所有必填构造参数，以匹配当前模型实现）
    print("Evaluating best checkpoint on TEST...")
    best_model = MFMEForTokenClassification.from_pretrained(
        best_path,
        num_labels=len(label2id),
        lexicon_size=lex_size, bigram_size=bigram_size, pos_size=pos_size, type_size=type_size,
        lex_dim=args.lex_dim, bigram_dim=args.bigram_dim, pos_dim=args.pos_dim, type_dim=args.type_dim,
        mem_hidden=args.mem_hidden
    ).to(device)

    test_metrics, test_report = evaluate(best_model, test_loader, id2label, device)
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
    ap.add_argument("--bigram_json", type=str, default="./feats/bigram.json")
    ap.add_argument("--pos_json", type=str, default="./feats/pos_vocab.json")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)  # 建议先 1e-5 稳住
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    # memory & features
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lex_dim", type=int, default=100)
    ap.add_argument("--bigram_dim", type=int, default=50)
    ap.add_argument("--pos_dim", type=int, default=16)
    ap.add_argument("--type_dim", type=int, default=8)
    ap.add_argument("--mem_hidden", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
