"""
Twinscribe — mental health detector (Streamlit UI).
Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import html
import warnings

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")

import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chatbot_inference import load_label_map, response_for_severity, run_inference

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "approach_a_bert_model")

APP_NAME = "Twinscribe"
APP_TITLE = "Mental health detector"
APP_TAGLINE = "BERT text triage · three severity bands · demo only, not a medical device"

# (display name, bar hex, verdict glow on dark inset)
_SEVERITY_VIS = {
    "No Symptoms": ("OK · steady", "#4ade80", "#22c55e"),
    "Mild/Moderate": ("Uneasy · worth attention", "#fbbf24", "#f59e0b"),
    "Severe": ("Urgent · act on it", "#fb7185", "#f43f5e"),
}


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

          html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif !important; }

          /* Wider column, vertically centered in the viewport */
          section[data-testid="stMain"] .block-container,
          .main .block-container {
            max-width: min(92vw, 920px) !important;
            width: 100% !important;
            min-height: 88vh !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            padding-top: clamp(1rem, 4vh, 2.5rem) !important;
            padding-bottom: clamp(1rem, 4vh, 2.5rem) !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
            box-sizing: border-box !important;
          }
          .app-brand {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: #c4a77d;
            margin: 0 0 0.35rem 0;
          }
          .app-title {
            font-family: 'Fraunces', serif;
            font-size: clamp(1.85rem, 4vw, 2.35rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #faf8f5;
            margin: 0 0 0.35rem 0;
            line-height: 1.12;
          }
          .app-sub {
            font-size: clamp(0.78rem, 1.6vw, 0.88rem);
            letter-spacing: 0.04em;
            color: #9c9690;
            margin: 0 0 1rem 0;
            line-height: 1.45;
            max-width: 42rem;
          }
          .status-pill {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(196,167,125,0.15);
            color: #d4c4a8;
            border: 1px solid rgba(196,167,125,0.35);
            margin-bottom: 0.85rem;
          }
          div[data-testid="stVerticalBlock"] > div:has(> div > textarea) {
            margin-bottom: 0.35rem !important;
          }
          .result-paper {
            margin-top: 0.65rem;
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 16px 56px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.12);
          }
          .result-paper .cream {
            background: linear-gradient(165deg, #faf7f2 0%, #f0e8de 100%);
            color: #141218;
            padding: 14px 20px 16px;
            font-size: 0.92rem;
            line-height: 1.4;
          }
          .result-paper .cream .echo {
            opacity: 0.55;
            font-size: 0.82rem;
            margin-bottom: 8px;
            font-style: italic;
          }
          .result-paper .inset-dark {
            background: radial-gradient(ellipse 120% 100% at 20% 0%, #1e1a24 0%, #0c0a0f 72%);
            color: #f5f0ea;
            padding: 22px 24px 24px;
            text-align: center;
          }
          .result-paper .micro {
            font-size: 0.68rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: rgba(245,240,234,0.45);
            margin-bottom: 8px;
          }
          .result-paper .verdict {
            font-family: 'Fraunces', serif;
            font-size: clamp(1.85rem, 3.5vw, 2.35rem);
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 6px;
          }
          .result-paper .tagline {
            font-size: 0.88rem;
            opacity: 0.75;
            margin-bottom: 4px;
          }
          .result-paper .conf {
            font-size: 0.8rem;
            opacity: 0.5;
          }
          .result-paper .bars {
            background: #faf7f2;
            padding: 14px 20px 16px;
            border-top: 1px solid rgba(20,18,24,0.08);
          }
          .pbar-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
            font-size: 0.82rem;
            color: #2d2933;
          }
          .pbar-row:last-child { margin-bottom: 0; }
          .pbar-lbl { width: 132px; flex-shrink: 0; font-weight: 600; }
          .pbar-track {
            flex: 1;
            height: 9px;
            border-radius: 99px;
            background: rgba(20,18,24,0.08);
            overflow: hidden;
          }
          .pbar-fill { height: 100%; border-radius: 99px; transition: width 0.35s ease; }
          .pbar-pct { width: 44px; text-align: right; font-variant-numeric: tabular-nums; opacity: 0.75; }
          .result-paper .reply {
            background: #f7f3ed;
            color: #1c1a1f;
            padding: 14px 20px;
            font-size: 0.88rem;
            line-height: 1.5;
            border-top: 1px solid rgba(20,18,24,0.06);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_bundle(model_dir: str):
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Not a directory: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = load_label_map(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.to(device)
    model.eval()
    return model, tokenizer, device, label_map["id2label"]


def _render_result(user_text: str, label: str, confidence: float, probs: dict, order: list[str]) -> None:
    tag, _bar_c, glow_c = _SEVERITY_VIS.get(label, ("—", "#94a3b8", "#64748b"))
    esc_user = html.escape(user_text[:280] + ("…" if len(user_text) > 280 else ""))
    esc_label = html.escape(label)
    esc_tag = html.escape(tag)

    bar_rows = []
    for name in order:
        pct = 100.0 * probs.get(name, 0.0)
        _, nc, _ = _SEVERITY_VIS.get(name, ("", "#94a3b8", "#64748b"))
        short = name.replace("/", "·")
        bar_rows.append(
            f'<div class="pbar-row">'
            f'<span class="pbar-lbl">{html.escape(short)}</span>'
            f'<div class="pbar-track"><div class="pbar-fill" style="width:{pct:.1f}%;background:{nc};"></div></div>'
            f'<span class="pbar-pct">{pct:.0f}%</span>'
            f"</div>"
        )

    reply = html.escape(response_for_severity(label, confidence))

    st.markdown(
        f"""
        <div class="result-paper">
          <div class="cream">
            <div class="echo">“{esc_user}”</div>
          </div>
          <div class="inset-dark">
            <div class="micro">read as</div>
            <div class="verdict" style="color:{glow_c}; text-shadow: 0 0 28px {glow_c}55;">{esc_label}</div>
            <div class="tagline">{esc_tag}</div>
            <div class="conf">{confidence:.0%} assigned to this bucket</div>
          </div>
          <div class="bars">{"".join(bar_rows)}</div>
          <div class="reply">{reply}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Twinscribe — Mental health detector",
        page_icon="◎",
        layout="centered",
    )
    _inject_style()

    with st.expander("setup", expanded=False):
        model_dir = st.text_input("Weights folder", value=DEFAULT_MODEL_DIR)
        max_length = st.number_input("Max tokens", 32, 512, 128, 32)

    try:
        model, tokenizer, device, id2label = load_model_bundle(model_dir)
    except Exception as e:
        st.error("Model failed to load")
        st.caption(str(e))
        st.stop()

    order = [id2label[i] for i in sorted(id2label.keys())]

    st.markdown(f'<p class="app-brand">{html.escape(APP_NAME)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-title">{html.escape(APP_TITLE)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-sub">{html.escape(APP_TAGLINE)}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-pill">{device.type.upper()} · {" · ".join(order)}</span>',
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "msg",
        height=128,
        placeholder="Paste or type one message — mood, worry, sleep, hope, anything on your mind…",
        label_visibility="collapsed",
    )
    go = st.button("Run detector", type="primary", use_container_width=True)

    if go and user_text.strip():
        label, confidence, probs = run_inference(
            user_text, model, tokenizer, device, id2label, max_length=int(max_length)
        )
        if not label:
            st.warning("Empty after cleaning.")
        else:
            st.session_state["u"] = user_text.strip()
            st.session_state["lab"] = label
            st.session_state["conf"] = confidence
            st.session_state["pr"] = probs

    if st.session_state.get("lab"):
        _render_result(
            st.session_state["u"],
            st.session_state["lab"],
            st.session_state["conf"],
            st.session_state["pr"],
            order,
        )

    st.caption(
        "Crisis: use emergency services or a helpline. Trained on public text; errors happen."
    )


if __name__ == "__main__":
    main()
