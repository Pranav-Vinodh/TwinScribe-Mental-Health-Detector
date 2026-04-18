# TwinScribe — Mental Health Detector

**Repository:** [github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector](https://github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector)

BERT-based **Approach A** text triage: three severity bands (No symptoms · Mild/Moderate · Severe), trained on the Kaggle mental health text dataset. Includes a Colab-oriented training notebook, a CLI, and a **Streamlit** UI (**Twinscribe**) with an in-session stateful chatbot that asks follow-up questions and uses prior turns for continuity.

> Demo / coursework — not a medical device.

## Relation to the base paper (IEEE Access “Digital Twin” dialogue system)

This repo is a **course-style adaptation**, not a full replication of the paper’s stack.

| Scope | Rough alignment |
|--------|-------------------|
| **Full paper system** (E-DAIC + PHQ labels, Rasa NLU/Core with stories/rules, digital-twin feedback loop, webchat deployment, formal usability study) | **~20–30%** — we do not ship Rasa, E-DAIC, PHQ integration, or that evaluation pipeline. |
| **“BERT drives severity-aware user guidance”** slice | **~45–55%** — we fine-tune BERT on public Reddit-style text with mapped severity bands and use it in a **Streamlit** chatbot that picks **follow-up prompts** from the predicted class (a lightweight mimic of “classification → feedback,” not Rasa dialogue policies). |

The Streamlit app is explicitly a **Streamlit mimic** of that idea: **not** Rasa, **not** the paper’s clinical authoring workflow, **not** a regulated medical device.

## Quick start (inference + UI)

1. Python 3.10+ recommended.
2. Place your fine-tuned folder next to this repo (default name `approach_a_bert_model/`) — produced by `approach_a_bert_training.ipynb` — or set `MODEL_DIR`.
3. Install and run Streamlit:

```bash
pip install -r requirements_chatbot.txt
streamlit run streamlit_app.py
```

CLI:

```bash
python chatbot_inference.py --model_dir ./approach_a_bert_model
```

## Chat UX behavior (current)

- The chat now shows only the direct assistant reply by default.
- Technical diagnostics are collapsed into expanders:
  - `Model score currently`: mapped intent, confidence, and per-class scores.
  - `Model/debug summary`: concise NLU mapping details.
  - `Conversation trace`: context and follow-up selection path for the turn.
- Assistant follow-up prompts use anti-repeat selection to reduce immediate repetition across same-intent turns.
- Lightweight topic cues from user text (for example sleep, support, work/study, safety language) steer follow-up prompt choice without changing model classification outputs.
- Safety guidance remains visible in every assistant response, and severe classification keeps explicit crisis-escalation wording.

## Training

Open `approach_a_bert_training.ipynb` in **Google Colab** (GPU), set `CSV_PATH` to `mental_heath_unbanlanced.csv`, run all cells, then download the saved model folder.

## Data & references

- `mental_heath_unbanlanced.csv` — Kaggle mental health text classification data.
- Reference PDFs in the repo: base IEEE Access paper (Digital Twin dialogue system).

