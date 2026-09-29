"""Reload the saved checkpoint and export metrics, errors and plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from common import ROOT, TextDataset, evaluate, read_rows, save_json
from inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "checkpoints/comment_classify/model_best"))
    parser.add_argument("--data", default=str(ROOT / "data/dev.txt"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predictor = Predictor(args.model, args.device)
    labels = dict(sorted(predictor.model.config.id2label.items()))
    rows = read_rows(args.data, labels)
    loader = DataLoader(TextDataset(rows, predictor.tokenizer, predictor.max_length), batch_size=16,
                        collate_fn=DataCollatorWithPadding(predictor.tokenizer, pad_to_multiple_of=8))
    metrics, predicted, confidence = evaluate(predictor.model, loader, predictor.device, labels)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / "evaluation.json", {"model": args.model, "data": args.data, **metrics})
    with (report_dir / "dev_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text", "true_label", "predicted_label", "confidence", "correct"])
        for row, pred, conf in zip(rows, predicted, confidence):
            writer.writerow([row["text"], labels[row["label"]], labels[pred], conf, row["label"] == pred])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    matrix = metrics["confusion_matrix"]
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=list(labels.values()),
           yticklabels=list(labels.values()), xlabel="预测类别", ylabel="真实类别", title="BERT-CLS 验证集混淆矩阵")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > max(map(max, matrix))/2 else "black")
    fig.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    history_file = report_dir / "history.json"
    if history_file.exists():
        history = json.loads(history_file.read_text(encoding="utf-8"))
        epochs = [r["epoch"] for r in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train loss")
        axes[0].plot(epochs, [r["dev_loss"] for r in history], label="Validation loss")
        axes[1].plot(epochs, [r["accuracy"] for r in history], label="Validation accuracy")
        axes[1].plot(epochs, [r["macro_f1"] for r in history], label="Validation macro F1")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.savefig(report_dir / "training_curves.png", dpi=160)
        plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
