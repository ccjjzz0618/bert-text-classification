"""Download the exact upstream Chinese BERT weights (safetensors only)."""
from huggingface_hub import snapshot_download
from common import ROOT, save_json

MODEL_ID = "google-bert/bert-base-chinese"
REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"

if __name__ == "__main__":
    destination = ROOT / "models/bert-base-chinese"
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=destination,
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "vocab.txt", "README.md"])
    save_json(destination / "source.json", {"model_id": MODEL_ID, "revision": REVISION})
    print(destination)
