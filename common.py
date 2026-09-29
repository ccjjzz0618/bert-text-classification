"""Shared data validation and evaluation for the BERT-CLS example."""
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_labels(path):
    labels = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            index, name = line.split(",", 1)
            index = int(index)
            if index in labels or not name.strip():
                raise ValueError(f"Duplicate/empty label in {path}: {line!r}")
            labels[index] = name.strip()
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError("Label IDs must be consecutive integers starting at zero")
    if len(set(labels.values())) != len(labels):
        raise ValueError("Label names must be unique")
    return dict(sorted(labels.items()))


def read_rows(path, labels):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            label, text = line.split("\t", 1)
            label = int(label)
            if label not in labels or not text.strip():
                raise ValueError("unknown label or empty text")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: expected label<TAB>text ({exc})") from exc
        rows.append({"label": label, "text": text.strip()})
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.encodings = tokenizer([r["text"] for r in rows], truncation=True,
                                   max_length=max_length, padding=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encodings.items()},
                "labels": self.rows[index]["label"]}


def select_device(name):
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run setup.ps1 to install GPU PyTorch, or use --device cpu.")
    return device


@torch.inference_mode()
def evaluate(model, loader, device, labels):
    was_training = model.training
    model.eval()
    gold, predicted, confidence = [], [], []
    total_loss = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        probs = output.logits.softmax(-1)
        conf, pred = probs.max(-1)
        gold.extend(batch["labels"].cpu().tolist())
        predicted.extend(pred.cpu().tolist())
        confidence.extend(conf.cpu().tolist())
        total_loss += output.loss.item() * len(pred)
    ids = list(labels)
    metrics = {
        "samples": len(gold), "loss": total_loss / len(gold),
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1_score(gold, predicted, labels=ids, average="macro", zero_division=0),
        "weighted_f1": f1_score(gold, predicted, labels=ids, average="weighted", zero_division=0),
        "classification_report": classification_report(gold, predicted, labels=ids,
            target_names=list(labels.values()), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=ids).tolist(),
    }
    model.train(was_training)
    return metrics, predicted, confidence
