"""Fine-tune Chinese BERT on the upstream eight-class comment dataset."""
import argparse
from collections import Counter
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from common import ROOT, TextDataset, evaluate, read_labels, read_rows, save_json, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/bert-base-chinese"))
    parser.add_argument("--train-path", default=str(ROOT / "data/train.txt"))
    parser.add_argument("--dev-path", default=str(ROOT / "data/dev.txt"))
    parser.add_argument("--labels", default=str(ROOT / "data/label_mapping.txt"))
    parser.add_argument("--output-dir", default=str(ROOT / "checkpoints/comment_classify"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 2 <= args.max_length <= 512:
        parser.error("epochs/batch-size must be positive; max-length must be 2..512")
    if args.learning_rate <= 0 or not 0 <= args.warmup_ratio <= 1:
        parser.error("learning-rate must be positive; warmup-ratio must be 0..1")
    out, reports = Path(args.output_dir), Path(args.report_dir)
    if (out / "model_best/config.json").exists():
        parser.error("Output already has a model. Choose a new --output-dir and --report-dir to preserve this run.")
    reports.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = select_device(args.device)
    labels = read_labels(args.labels)
    train_rows, dev_rows = read_rows(args.train_path, labels), read_rows(args.dev_path, labels)
    overlap = {r["text"] for r in train_rows} & {r["text"] for r in dev_rows}
    if overlap:
        raise ValueError(f"Train/dev text overlap: {len(overlap)}; split your data before training")
    if {r["label"] for r in train_rows} != set(labels):
        raise ValueError("Every label needs at least one training sample")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=labels,
        label2id={v: k for k, v in labels.items()}, attn_implementation="sdpa")
    model.to(device)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(TextDataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    dev_loader = DataLoader(TextDataset(dev_rows, tokenizer, args.max_length),
        batch_size=args.batch_size, collate_fn=collator, num_workers=0)
    no_decay = ("bias", "LayerNorm.weight")
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)],
         "weight_decay": 0.0}], lr=args.learning_rate)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {**vars(args), "torch": torch.__version__, "transformers": transformers.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "amp": use_amp, "train_samples": len(train_rows), "dev_samples": len(dev_rows),
        "train_label_counts": dict(Counter(r["label"] for r in train_rows)),
        "dev_label_counts": dict(Counter(r["label"] for r in dev_rows)), "text_overlap": len(overlap),
        "data_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in (args.train_path, args.dev_path, args.labels)}}
    save_json(reports / "run_config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    start = time.perf_counter()
    history, best_f1, best_epoch = [], -1.0, 0
    log = reports / "training.jsonl"
    log.write_text("", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            total_loss += loss.item() * len(batch["labels"])
            if step % 10 == 0:
                print(f"epoch {epoch:02d}/{args.epochs}, step {step}/{len(train_loader)}, loss {loss.item():.4f}", flush=True)
        metrics, _, _ = evaluate(model, dev_loader, device, labels)
        row = {"epoch": epoch, "train_loss": total_loss / len(train_rows),
               "dev_loss": metrics["loss"], "accuracy": metrics["accuracy"],
               "macro_f1": metrics["macro_f1"], "elapsed_seconds": time.perf_counter() - start}
        history.append(row)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            model.save_pretrained(out / "model_best", safe_serialization=True)
            tokenizer.save_pretrained(out / "model_best")
            save_json(out / "model_best/training_info.json", {"epoch": epoch, "max_length": args.max_length,
                       "macro_f1": best_f1, "seed": args.seed})
            save_json(reports / "best_dev_metrics.json", {"epoch": epoch, **metrics})
            print(f"Saved best checkpoint: epoch={epoch}, macro_f1={best_f1:.6f}", flush=True)
    save_json(reports / "history.json", history)
    save_json(reports / "training_summary.json", {"completed_epochs": args.epochs,
        "best_epoch": best_epoch, "best_macro_f1": best_f1,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
        "best_model": str(out / "model_best")})
    print("Training finished. Best model:", out / "model_best", flush=True)


if __name__ == "__main__":
    main()
