# TwinScribe — Mental Health Detector

**Repository:** [github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector](https://github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector)

TwinScribe is a BERT-based mental health text triage project with two user-facing modes:

- a command-line inference tool for quick label prediction
- a Streamlit chatbot UI that keeps in-session context, asks structured follow-up questions, and provides practical suggestions

The classifier predicts one of three severity bands:

- `No Symptoms`
- `Mild/Moderate`
- `Severe`

> Demo / coursework project only. This is **not** a medical device and does **not** provide diagnosis.

---

## What this project does

Given user-written text, TwinScribe:

1. cleans and tokenizes the text
2. runs a fine-tuned BERT sequence classifier
3. maps model probabilities to severity labels
4. generates supportive, safety-aware responses
5. in the Streamlit app, continues with a structured interview flow and periodic summaries/advice

The chatbot is intentionally designed as a **screening-style support assistant**, not a replacement for a clinician.

---

## Core features

- **Three-class severity inference** using a fine-tuned Transformer model
- **Per-class probability output** (`No Symptoms`, `Mild/Moderate`, `Severe`)
- **Stateful chatbot behavior** across turns in one session
- **Structured interview domains** (wellbeing, sleep/physical health, support, daily functioning, etc.)
- **Answer-aware continuity** (the bot uses the previous answer before asking the next question)
- **Conclusion + tips mode** after a few turns (summary of what was shared + actionable next steps)
- **Safety escalation language** retained for severe-risk interactions
- **Collapsed diagnostics UI** so normal users mainly see conversation text

---

## Repository structure

- `streamlit_app.py`  
  Main web app and conversational logic.
- `chatbot_inference.py`  
  Model loading, text preprocessing, classification inference, and severity framing.
- `approach_a_bert_training.ipynb`  
  Notebook used to train/export model artifacts.
- `requirements_chatbot.txt`  
  Python dependencies for running the app and inference.
- `mental_heath_unbanlanced.csv`  
  Dataset file used in the project workflow.
- `approach_a_bert_model/` (expected, generated externally)  
  Exported model directory with tokenizer/model weights and label map.

---

## Model artifacts expected

The runtime expects a model folder (default `approach_a_bert_model/`) containing files such as:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- tokenizer files (`tokenizer.json`, vocab, merges, etc. depending on tokenizer)
- `label_map.json` (optional; fallback mapping is used if missing)

You can set a custom location with the `MODEL_DIR` environment variable or UI setup input.

---

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements_chatbot.txt
```

---

## Run the Streamlit chatbot

```bash
streamlit run streamlit_app.py
```

In the app:

- open the `setup` expander
- confirm `Weights folder` points to your model directory
- send chat messages in the input box

---

## Run CLI inference

```bash
python chatbot_inference.py --model_dir ./approach_a_bert_model
```

This starts an interactive terminal chat loop and prints model-aligned responses.

---

## Chatbot conversation behavior

The current chatbot flow is:

1. classify each user message (`No Symptoms`, `Mild/Moderate`, `Severe`)
2. continue a structured interview with domain-aware questions
3. acknowledge and use the latest answer context
4. after a few Q/A turns, produce a concise conclusion:
   - what it heard so far
   - suggested next steps
   - guidance on when to seek professional support

For severe-risk cases, safety-first wording is prioritized.

---

## Streamlit UI behavior

- Main chat shows only the assistant/user messages.
- Diagnostics are available in collapsible sections:
  - `Model score currently` (top label + class probabilities)
  - `Model/debug summary` (intent mapping and confidence details)
  - `Conversation trace` (internal flow indicators)

This keeps the primary UX conversational while preserving transparency for debugging.

---

## Training notes

Use `approach_a_bert_training.ipynb` (typically in Google Colab with GPU):

1. set `CSV_PATH` to `mental_heath_unbanlanced.csv`
2. run training/evaluation/export cells
3. download exported model folder
4. place it in this repo as `approach_a_bert_model/` (or set `MODEL_DIR`)

---

## Safety and limitations

- This tool is for **screening-style support and educational use**.
- Predictions can be wrong (false positives/false negatives).
- Output quality depends on training data quality and coverage.
- It does not perform clinical diagnosis.
- In any immediate safety crisis, users should contact emergency services or a local crisis line.

---

## License / usage context

This repository is maintained as a student/coursework-style project.  
Use responsibly, and avoid deploying as a standalone clinical decision system without formal clinical validation and governance.

