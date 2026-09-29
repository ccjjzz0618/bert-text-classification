"""Predict Chinese labels from a locally saved BERT classifier."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import ROOT, save_json, select_device


class Predictor:
    def __init__(self, model_path=ROOT / "checkpoints/comment_classify/model_best", device="auto"):
        self.device = select_device(device)
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True).to(self.device).eval()
        info = Path(model_path) / "training_info.json"
        self.max_length = json.loads(info.read_text(encoding="utf-8"))["max_length"] if info.exists() else 128

    @torch.inference_mode()
    def predict(self, texts, batch_size=16):
        if batch_size < 1 or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Use nonempty texts and a positive batch_size")
        results = []
        for offset in range(0, len(texts), batch_size):
            batch_texts = texts[offset:offset + batch_size]
            inputs = self.tokenizer(batch_texts, truncation=True, max_length=self.max_length,
                                    padding=True, return_tensors="pt").to(self.device)
            probs = self.model(**inputs).logits.softmax(-1).cpu()
            for text, scores in zip(batch_texts, probs):
                index = scores.argmax().item()
                results.append({"text": text, "label_id": index,
                    "label": self.model.config.id2label[index], "confidence": scores[index].item(),
                    "scores": {self.model.config.id2label[i]: s.item() for i, s in enumerate(scores)}})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "checkpoints/comment_classify/model_best"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text", nargs="+")
    parser.add_argument("--input-file", help="UTF-8 file, one text per line")
    parser.add_argument("--output", help="Write predictions to JSON")
    args = parser.parse_args()
    if args.text and args.input_file:
        parser.error("Choose --text or --input-file")
    if args.output and not (args.text or args.input_file):
        parser.error("--output requires --text or --input-file")
    predictor = Predictor(args.model, args.device)
    if args.text or args.input_file:
        texts = args.text or [s.strip() for s in Path(args.input_file).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if not texts:
            parser.error("Input file is empty")
        result = predictor.predict(texts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.output:
            save_json(args.output, result)
    else:
        print("模型已加载。输入中文评论并回车，输入 q 退出。")
        while True:
            try:
                text = input("\n文本 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"q", "quit", "exit"}:
                break
            if text:
                result = predictor.predict([text])[0]
                print(f"类别：{result['label']}  分数：{result['confidence']:.2%}")


if __name__ == "__main__":
    main()
