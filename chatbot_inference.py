#!/usr/bin/env python3
"""
Simple mental-health severity chatbot using a fine-tuned BERT classifier (Approach A).
Point MODEL_DIR to the folder saved by the training notebook (contains config.json, pytorch_model.bin or model.safetensors, tokenizer files, label_map.json).
"""
from __future__ import annotations

import warnings

# Hugging Face `transformers` lazy-imports hundreds of model submodules; Python 3.12+ warns on each.
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import transformers

transformers.utils.logging.set_verbosity_error()
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_label_map(model_dir: Path) -> dict:
    path = model_dir / "label_map.json"
    if not path.exists():
        return {
            "id2label": {0: "No Symptoms", 1: "Mild/Moderate", 2: "Severe"},
            "label2id": {"No Symptoms": 0, "Mild/Moderate": 1, "Severe": 2},
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return {"id2label": id2label, "label2id": data["label2id"]}


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_inference(
    raw_text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    id2label: dict[int, str],
    max_length: int = 128,
) -> tuple[str, float, dict[str, float]]:
    """Returns predicted label name, confidence for that label, and probabilities for all classes."""
    text = clean_text(raw_text)
    if not text:
        empty = {id2label[i]: 0.0 for i in sorted(id2label.keys())}
        return "", 0.0, empty

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())
    confidence = float(probs[pred_id].item())
    label = id2label.get(pred_id, str(pred_id))
    prob_by_label = {id2label[i]: float(probs[i].item()) for i in range(len(probs))}
    return label, confidence, prob_by_label


def response_for_severity(label: str, confidence: float) -> str:
    """Template responses aligned with a screening / digital-twin style dialogue (not clinical advice)."""
    c = f" (model confidence about {confidence:.0%})"
    if label == "No Symptoms":
        return (
            "From this message alone, the model does not flag strong mental-health distress signals."
            + c
            + " If anything still feels off, talking to someone you trust or a professional is always reasonable."
        )
    if label == "Mild/Moderate":
        return (
            "The model suggests mild-to-moderate distress themes in what you wrote."
            + c
            + " Consider self-care, reaching out to supportive people, and—if symptoms persist or worsen—speaking with a qualified mental health professional."
        )
    return (
        "The model flags this message in the highest concern category we use in this project."
        + c
        + " If you are in immediate danger or thinking about harming yourself, please contact local emergency services or a suicide/crisis line right away. You deserve support."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mental health severity chatbot (BERT Approach A)")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "./approach_a_bert_model"),
        help="Directory with saved model + tokenizer + label_map.json",
    )
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        print(f"MODEL_DIR not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(model_dir)
    id2label = label_map["id2label"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    print("Approach A chatbot — type 'quit' to exit.")
    print("This is a course demo, not medical diagnosis.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            break

        label, confidence, _ = run_inference(
            user, model, tokenizer, device, id2label, max_length=args.max_length
        )
        if not label:
            print("Bot: Please enter some text.")
            continue
        print("Bot:", response_for_severity(label, confidence))
        print()


if __name__ == "__main__":
    main()
