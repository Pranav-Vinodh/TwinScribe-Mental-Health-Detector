# TwinScribe — Mental Health Detector

**Repository:** [github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector](https://github.com/Pranav-Vinodh/TwinScribe-Mental-Health-Detector)

BERT-based **Approach A** text triage: three severity bands (No symptoms · Mild/Moderate · Severe), trained on the Kaggle mental health text dataset. Includes a Colab-oriented training notebook, a CLI, and a **Streamlit** UI (**Twinscribe**).

> Demo / coursework — not a medical device.

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

## Training

Open `approach_a_bert_training.ipynb` in **Google Colab** (GPU), set `CSV_PATH` to `mental_heath_unbanlanced.csv`, run all cells, then download the saved model folder.

## Data & references

- `mental_heath_unbanlanced.csv` — Kaggle mental health text classification data.
- Reference PDFs in the repo: base IEEE Access paper (Digital Twin dialogue system).

