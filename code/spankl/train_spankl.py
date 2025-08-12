import os, json, math, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertConfig, BertTokenizerFast, set_seed, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from conll_utils import read_conll, build_label_map
from dataset_spankl import SpanKLDataset, CollateSpan
from model_spankl import SpanKLForNER
from spans import greedy_select_non_overlapping, spans_to_tags


def estimate_type_prior(train_labels, label2id):
    # 统计各实体类型（非 'O'）的span数量
    counts = {i:0 for i in range(len(label2id))}
    for tags in train_labels:
        prev = "O"
        for t in tags + ["O"]:
            if t != prev:
                if prev != "O":
                    counts[label2id[prev]] += 1
                prev = t
    total = sum(v for k,v in counts.items() if k!=0)
    C = len(label2id) - 1
    if total == 0 or C <= 0:
        return np.ones(max(1,C)) / max(1,C)
    p = np.array([counts[i] for i in range(1,len(label2id))], dtype=np.float64)
    p = p / p.sum()
    return p

def evaluate(model, loader, id2label, device, threshold=0.5):
    model.eval()
    all_true, all_pred = [], []
    token_total, token_correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            for k in ["input_ids","attention_mask","token_type_ids","tok2char_index"]:
                batch[k] = batch[k].to(device)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        span_positions=batch["span_positions"].to(device),
                        span_batch_idx=batch["span_batch_idx"].to(device))
            logits = out["logits"]  # [T,C]
            probs = torch.softmax(logits, dim=-1)
            scores, preds = probs.max(dim=-1)  # [T]

            # 汇总到句子
            spans_by_sent = {}
            T = preds.size(0)
            for i in range(T):
                bi = int(batch["span_batch_idx"][i].cpu().item())
                if bi not in spans_by_sent: spans_by_sent[bi] = []
                lab_id = int(preds[i].cpu().item())
                if lab_id == 0:  # 'O'
                    continue
                sc = float(scores[i].cpu().item())
                if sc < threshold:
                    continue
                st = int(batch["span_positions"][i,0].cpu().item())
                ed = int(batch["span_positions"][i,1].cpu().item())
                lab = id2label[lab_id]
                spans_by_sent[bi].append((st, ed, lab, sc))

            B = batch["input_ids"].size(0)
            for bi in range(B):
                true_tags = batch["orig_labels"][bi]
                n_chars = int(batch["n_chars"][bi])
                # 贪心去重
                cand = spans_by_sent.get(bi, [])
                cand.sort(key=lambda x: x[3], reverse=True)
                selected = greedy_select_non_overlapping(cand)
                # token->char 映射
                tok2char = batch["tok2char_index"][bi].cpu().numpy().tolist()
                pred_char_spans = []
                for (st_tok, ed_tok, lab, sc) in selected:
                    if st_tok < 0 or ed_tok >= len(tok2char): continue
                    cs = tok2char[st_tok]; ce = tok2char[ed_tok]
                    if cs == -1 or ce == -1: continue
                    if cs > ce: cs, ce = ce, cs
                    pred_char_spans.append((cs, ce, lab))
                pred_tags = spans_to_tags(n_chars, [(s,e,l) for (s,e,l) in pred_char_spans])

                token_total += len(true_tags)
                token_correct += sum([t==p for t,p in zip(true_tags, pred_tags)])
                all_true.append(true_tags)
                all_pred.append(pred_tags)

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
    # 数据与标签表
    train_s, train_l = read_conll(os.path.join(args.data_dir, "train.txt"))
    dev_s, dev_l = read_conll(os.path.join(args.data_dir, "dev.txt"))
    test_s, test_l = read_conll(os.path.join(args.data_dir, "test.txt"))
    lm = build_label_map(train_l, dev_l, test_l)
    label2id, id2label = lm["label2id"], {int(i):l for i,l in lm["id2label"].items()}

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)

    # 类型先验（排除 'O'）
    p_prior = estimate_type_prior(train_l, label2id)
    np.save(os.path.join(args.out_dir, "type_prior.npy"), p_prior)

    # 数据集 & DataLoader
    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained)
    collate = CollateSpan(pad_id=tokenizer.pad_token_id)

    train_ds = SpanKLDataset(args.data_dir, "train", args.pretrained, label2id,
                             max_len=args.max_len, max_span_len=args.max_span_len,
                             neg_ratio=args.neg_ratio, max_negs=args.max_negs)
    dev_ds   = SpanKLDataset(args.data_dir, "dev", args.pretrained, label2id,
                             max_len=args.max_len, max_span_len=args.max_span_len,
                             neg_ratio=args.neg_ratio, max_negs=args.max_negs)
    test_ds  = SpanKLDataset(args.data_dir, "test", args.pretrained, label2id,
                             max_len=args.max_len, max_span_len=args.max_span_len,
                             neg_ratio=args.neg_ratio, max_negs=args.max_negs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # 模型
    config = BertConfig.from_pretrained(
        args.pretrained,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    # 把自定义超参写入 config，便于 from_pretrained 重建
    config.max_span_len = int(args.max_span_len)

    model = SpanKLForNER.from_pretrained(
        args.pretrained,
        config=config,  # 训练时仍然用预训练权重初始化
        num_labels=len(label2id),
        max_span_len=args.max_span_len
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
    t_total = math.ceil(len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*args.warmup), t_total)

    # 训练
    best_f1, best_path = -1.0, None
    p_prior_t = torch.tensor(p_prior, dtype=torch.float32, device=device)  # [C-1]
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            for k in ["input_ids","attention_mask","token_type_ids"]:
                batch[k] = batch[k].to(device)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        span_positions=batch["span_positions"].to(device),
                        span_batch_idx=batch["span_batch_idx"].to(device),
                        span_labels=batch["span_labels"].to(device))
            ce_loss = out["loss"]

            # KL：仅对正样本（非'O'）的类型分布做约束
            logits = out["logits"]
            labels = batch["span_labels"].to(device)
            pos_mask = labels != 0
            kl_loss = torch.tensor(0.0, device=device)
            if pos_mask.any() and p_prior_t.numel() > 0:
                logits_pos = logits[pos_mask][:,1:]            # 去掉 'O'
                q = torch.softmax(logits_pos, dim=-1).mean(dim=0)
                kl_loss = torch.sum(q * (torch.log(q + 1e-8) - torch.log(p_prior_t + 1e-8)))
            loss = ce_loss + args.kl_alpha * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += float(loss.item())
            if step % args.log_steps == 0:
                print(f"Epoch {epoch} Step {step} | loss={total_loss/args.log_steps:.4f} (ce={float(ce_loss.item()):.4f}, kl={float(kl_loss.item()):.4f})")
                total_loss = 0.0

        # DEV
        dev_metrics, dev_report = evaluate(model, dev_loader, id2label, device, threshold=args.decode_threshold)
        print(f"[DEV] epoch={epoch} f1={dev_metrics['f1']:.4f} p={dev_metrics['precision']:.4f} r={dev_metrics['recall']:.4f} acc={dev_metrics['token_acc']:.4f}")
        with open(os.path.join(args.out_dir, f"dev_report_epoch{epoch}.txt"), "w", encoding="utf-8") as f:
            f.write(dev_report)
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_path = os.path.join(args.out_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            # 保存 tokenizer，方便下游直接加载同一套词表

            BertTokenizerFast.from_pretrained(args.pretrained).save_pretrained(best_path)
            with open(os.path.join(best_path, "dev_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, ensure_ascii=False, indent=2)

    # ---- TEST 加载处改成无需额外参数 ----
    print("Evaluating best checkpoint on TEST...")
    best_model = SpanKLForNER.from_pretrained(best_path).to(device)
    test_metrics, test_report = evaluate(best_model, test_loader, id2label, device, threshold=args.decode_threshold)
    print(test_report)
    with open(os.path.join(args.out_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./run")
    parser.add_argument("--pretrained", type=str, default="bert-base-chinese")
    parser.add_argument("--max_len", type=int, default=160)
    parser.add_argument("--max_span_len", type=int, default=8)
    parser.add_argument("--neg_ratio", type=int, default=3)
    parser.add_argument("--max_negs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--decode_threshold", type=float, default=0.5)
    parser.add_argument("--kl_alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
