"""
Twinscribe — mental health detector (Streamlit UI).
Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import html
import random
import re
import warnings

warnings.filterwarnings("ignore", message=r"Accessing `__path__` from")

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chatbot_inference import load_label_map, response_for_severity, run_inference

DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR", "approach_a_bert_model")

APP_NAME = "Twinscribe"
APP_TITLE = "Mental health detector"
APP_TAGLINE = (
    "Context-aware in-session chat + BERT signal each turn · demo only, not a medical device"
)


def _init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot_state" not in st.session_state:
        st.session_state.chatbot_state = {
            "pending_question": None,
            "pending_domain": None,
            "asked_questions": [],
            "prev_label": None,
            "turn_count": 0,
            "domain_turns": {},
            "insights": [],
            "answered_count": 0,
        }


def _clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.chatbot_state = {
        "pending_question": None,
        "pending_domain": None,
        "asked_questions": [],
        "prev_label": None,
        "turn_count": 0,
        "domain_turns": {},
        "insights": [],
        "answered_count": 0,
    }


def _classification_summary(label: str, confidence: float, probs: dict[str, float], order: list[str]) -> str:
    intent = label.lower().replace("/", "_").replace(" ", "_")
    return "\n".join(
        [
            f"- intent: `{intent}` (from severity `{label}`)",
            f"- **confidence:** {confidence:.0%}",
            f"- class scores: " + " · ".join(f"`{k}` {probs.get(k, 0):.0%}" for k in order),
        ]
    )


def _detect_topics(text: str) -> set[str]:
    t = text.lower()
    topics: set[str] = set()
    lex = {
        "sleep": ["sleep", "slept", "insomnia", "awake", "tired", "exhausted", "restless"],
        "anxiety": ["anxious", "anxiety", "panic", "overthinking", "worried", "worry"],
        "mood": ["sad", "down", "hopeless", "empty", "low", "cry"],
        "support": ["friend", "family", "alone", "lonely", "talk", "support"],
        "work": ["work", "job", "office", "deadline", "study", "college", "exam", "class"],
        "safety": ["harm", "suicide", "kill myself", "end it", "unsafe"],
    }
    for topic, words in lex.items():
        if any(w in t for w in words):
            topics.add(topic)
    return topics


def _recent_user_messages(messages: list[dict], limit: int = 4) -> list[str]:
    turns = [str(m.get("content") or "").strip() for m in messages if m.get("role") == "user"]
    return [t for t in turns if t][-limit:]


_INTERVIEW_DOMAINS = [
    "wellbeing",
    "physical_health",
    "hope_hopelessness",
    "self_perception",
    "relationships_support",
    "activity_functioning",
    "choice_control",
]

_INTERVIEW_QUESTIONS: list[dict[str, Any]] = [
    {"id": "wellbeing_overall", "domain": "wellbeing", "tags": {"mood", "anxiety"}, "text": "How would you describe your overall emotional state today in a few words?"},
    {"id": "wellbeing_change", "domain": "wellbeing", "tags": {"mood", "anxiety"}, "text": "Compared with last week, does your mood feel better, worse, or about the same?"},
    {"id": "sleep_quality", "domain": "physical_health", "tags": {"sleep"}, "text": "How has your sleep been recently: trouble falling asleep, staying asleep, or waking too early?"},
    {"id": "energy_appetite", "domain": "physical_health", "tags": {"sleep", "mood"}, "text": "Have you noticed changes in your energy or appetite lately?"},
    {"id": "hope_future", "domain": "hope_hopelessness", "tags": {"mood"}, "text": "When you think about the next few weeks, do you feel hopeful, uncertain, or mostly stuck?"},
    {"id": "positive_moment", "domain": "hope_hopelessness", "tags": {"mood"}, "text": "Was there any moment recently that felt even slightly positive or relieving?"},
    {"id": "self_worth", "domain": "self_perception", "tags": {"mood"}, "text": "Have you been feeling more self-critical or blaming yourself more than usual?"},
    {"id": "confidence_change", "domain": "self_perception", "tags": {"mood", "work"}, "text": "Has your confidence in handling daily responsibilities changed recently?"},
    {"id": "support_contact", "domain": "relationships_support", "tags": {"support"}, "text": "Who do you feel most comfortable talking to when things get heavy?"},
    {"id": "isolation_level", "domain": "relationships_support", "tags": {"support", "mood"}, "text": "Have you been withdrawing from people more than usual, or still staying somewhat connected?"},
    {"id": "daily_function", "domain": "activity_functioning", "tags": {"work", "mood"}, "text": "How much is this affecting your work, study, or day-to-day routine right now?"},
    {"id": "concentration_focus", "domain": "activity_functioning", "tags": {"work", "anxiety"}, "text": "Have focus and concentration become harder recently?"},
    {"id": "control_feeling", "domain": "choice_control", "tags": {"anxiety", "mood"}, "text": "Do you feel you have some control over your next step, or mostly feel overwhelmed?"},
    {"id": "small_step", "domain": "choice_control", "tags": {"work", "mood"}, "text": "What is one small, realistic step you could take today for your wellbeing?"},
    {"id": "safety_now", "domain": "wellbeing", "tags": {"safety"}, "text": "Are you safe right now, and is someone nearby or reachable who can stay with you?"},
    {"id": "urgent_support", "domain": "relationships_support", "tags": {"safety", "support"}, "text": "Would you like help contacting a crisis helpline or local emergency support now?"},
]


def _next_domain(state: dict) -> str:
    turns = state.get("domain_turns", {})
    for d in _INTERVIEW_DOMAINS:
        if int(turns.get(d, 0)) < 2:
            return d
    return _INTERVIEW_DOMAINS[0]


def _pick_next_question(topics: set[str], label: str, state: dict) -> tuple[str, str, str]:
    asked = set(state.get("asked_questions", []))
    current_domain = _next_domain(state)
    unasked = [q for q in _INTERVIEW_QUESTIONS if q["id"] not in asked]
    if not unasked:
        unasked = _INTERVIEW_QUESTIONS

    if label == "Severe":
        safety_unasked = [q for q in unasked if "safety" in q["tags"]]
        if safety_unasked:
            q = random.choice(safety_unasked)
            return q["id"], q["text"], q["domain"]

    topic_matched = [q for q in unasked if q["tags"] & topics]
    if topic_matched:
        domain_matched = [q for q in topic_matched if q["domain"] == current_domain]
        q = random.choice(domain_matched or topic_matched)
        return q["id"], q["text"], q["domain"]

    domain_unasked = [q for q in unasked if q["domain"] == current_domain]
    if domain_unasked:
        q = random.choice(domain_unasked)
        return q["id"], q["text"], q["domain"]

    q = random.choice(unasked)
    return q["id"], q["text"], q["domain"]


def _reply_to_pending_question(pending: str | None, user_text: str) -> str | None:
    if not pending:
        return None
    low = user_text.lower()
    yes = bool(re.search(r"\b(yes|yeah|yep|sometimes|often|mostly)\b", low))
    no = bool(re.search(r"\b(no|not really|rarely|never)\b", low))
    if pending == "safety_now" or pending == "urgent_support":
        if no:
            return "Thank you for saying that clearly. Your safety matters most right now, so please contact emergency services or a crisis helpline immediately."
        if yes:
            return "I’m relieved you’re in a safer spot right now. Let’s keep you supported while this feels intense."
        return "Thank you for sharing that. Let’s keep focusing on immediate safety first."
    if pending in {"sleep_quality", "energy_appetite", "concentration_focus"}:
        return "That helps clarify the physical pattern around sleep, energy, and focus."
    if pending in {"support_contact", "isolation_level"}:
        return "Thanks for sharing your support situation, that context is important."
    if pending in {"daily_function", "small_step", "control_feeling"}:
        return "Got it, that gives useful context about daily functioning and what feels manageable."
    if pending in {"self_worth", "confidence_change", "hope_future", "positive_moment"}:
        return "Thank you for sharing that honestly. It helps me understand your emotional pattern better."
    if pending == "wellbeing_overall" or pending == "wellbeing_change":
        return "Thanks, that gives me a clearer picture of how things are feeling overall."
    if yes:
        return "Thanks, that helps."
    if no:
        return "Thanks for clarifying."
    return "Thanks for sharing that."


def _extract_insight(pending: str | None, user_text: str) -> str | None:
    if not pending:
        return None
    clean = " ".join(user_text.strip().split())
    if not clean:
        return None
    short = clean[:160] + ("..." if len(clean) > 160 else "")
    mapping = {
        "wellbeing_overall": f"Overall emotional state: {short}",
        "wellbeing_change": f"Recent mood trend: {short}",
        "sleep_quality": f"Sleep pattern reported: {short}",
        "energy_appetite": f"Energy/appetite change: {short}",
        "hope_future": f"Future outlook: {short}",
        "positive_moment": f"Recent positive moments: {short}",
        "self_worth": f"Self-perception concern: {short}",
        "confidence_change": f"Confidence in tasks: {short}",
        "support_contact": f"Support contact: {short}",
        "isolation_level": f"Social connection level: {short}",
        "daily_function": f"Daily functioning impact: {short}",
        "concentration_focus": f"Focus/concentration issue: {short}",
        "control_feeling": f"Sense of control: {short}",
        "small_step": f"Proposed small step: {short}",
        "safety_now": f"Immediate safety status: {short}",
        "urgent_support": f"Urgent help preference: {short}",
    }
    return mapping.get(pending, f"Response to {pending}: {short}")


def _tips_from_topics_and_label(topics: set[str], label: str) -> list[str]:
    tips: list[str] = []
    if "sleep" in topics:
        tips.append("Try a short sleep reset routine tonight: fixed wind-down time, no doom-scrolling 30 minutes before bed, and low light.")
    if "anxiety" in topics:
        tips.append("Use a 2-minute grounding cycle when anxiety spikes: slow exhale, name 5-4-3-2-1 sensory anchors, then resume one small task.")
    if "work" in topics:
        tips.append("Break your next responsibility into one 10-minute starter step; momentum usually helps more than waiting for motivation.")
    if "support" in topics:
        tips.append("Send one short check-in text to a trusted person today; brief connection can reduce isolation load.")
    if "mood" in topics:
        tips.append("Plan one low-effort restoring activity today (walk, shower, sunlight, or music) even if motivation is low.")

    if label == "Severe":
        tips.insert(0, "If you feel at risk of self-harm, contact local emergency services or a crisis helpline immediately and stay with someone trusted.")
    elif label == "Mild/Moderate":
        tips.append("If these symptoms continue for 2+ weeks or worsen, book time with a mental health professional or GP.")
    else:
        tips.append("Keep monitoring how you feel this week; early support is still useful even when distress signals are lower.")

    if not tips:
        tips = [
            "Keep your next step small and specific for today rather than trying to solve everything at once.",
            "If this continues to affect daily life, consider reaching out to a qualified mental health professional.",
        ]
    return tips[:3]


def _should_give_conclusion(state: dict, label: str) -> bool:
    answered = int(state.get("answered_count", 0))
    turns = int(state.get("turn_count", 0))
    if label == "Severe" and answered >= 1:
        return True
    return answered >= 3 and turns % 2 == 0


def _build_conclusion_block(state: dict, topics: set[str], label: str) -> str:
    insights = list(state.get("insights", []))
    recent = insights[-3:]
    summary_lines = "\n".join(f"- {item}" for item in recent) if recent else "- I have limited details so far, but I can still suggest practical next steps."
    tips = _tips_from_topics_and_label(topics, label)
    tips_lines = "\n".join(f"- {item}" for item in tips)
    return (
        "Here is what I’m hearing so far:\n"
        f"{summary_lines}\n\n"
        "Suggested next steps:\n"
        f"{tips_lines}\n\n"
        "If you want, we can keep going and build a clearer plan for the next 24 hours."
    )


def _build_chatbot_reply(
    user_text: str,
    label: str,
    confidence: float,
    probs: dict[str, float],
    order: list[str],
    messages: list[dict],
    state: dict,
) -> tuple[str, str, list[str], dict]:
    new_state = dict(state)
    new_state["turn_count"] = int(new_state.get("turn_count", 0)) + 1
    topics = _detect_topics(user_text)
    recent_users = _recent_user_messages(messages)
    trace = [
        "engine:clinical_interview_chatbot",
        f"context:user_turns={len(recent_users)}",
        f"context:topics={','.join(sorted(topics)) if topics else 'none'}",
    ]
    blocks: list[str] = []

    pending = new_state.get("pending_question")
    ack = _reply_to_pending_question(pending, user_text)
    if ack:
        blocks.append(ack)
        trace.append(f"context:answered={pending}")
        insight = _extract_insight(pending, user_text)
        if insight:
            gathered = list(new_state.get("insights", []))
            gathered.append(insight)
            new_state["insights"] = gathered[-12:]
        new_state["answered_count"] = int(new_state.get("answered_count", 0)) + 1

    prev_label = new_state.get("prev_label")
    is_first_turn = prev_label is None
    label_changed = prev_label != label
    if label == "Severe" or is_first_turn or label_changed:
        blocks.append(response_for_severity(label, confidence))
        trace.append("response:severity_framing")
    else:
        blocks.append("I hear you, and I’m tracking what you shared so far. Let’s build on that.")
        trace.append("response:context_ack")

    if len(recent_users) >= 2:
        blocks.append("I’m using what you shared across turns so we can keep this structured but still personal to your situation.")
        trace.append("context:multi_turn_reference")

    if _should_give_conclusion(new_state, label):
        blocks.append(_build_conclusion_block(new_state, topics, label))
        trace.append("response:conclusion_and_tips")
        new_state["pending_question"] = None
        new_state["pending_domain"] = None
        new_state["prev_label"] = label
        debug_text = _classification_summary(label, confidence, probs, order)
        return "\n\n".join(blocks), debug_text, trace, new_state

    q_id, question, q_domain = _pick_next_question(topics, label, new_state)
    blocks.append(question)
    trace.append(f"next_question:{q_id}")
    trace.append(f"next_domain:{q_domain}")
    asked = list(new_state.get("asked_questions", []))
    asked.append(q_id)
    new_state["asked_questions"] = asked[-12:]
    new_state["pending_question"] = q_id
    new_state["pending_domain"] = q_domain
    domain_turns = dict(new_state.get("domain_turns", {}))
    domain_turns[q_domain] = int(domain_turns.get(q_domain, 0)) + 1
    new_state["domain_turns"] = domain_turns
    new_state["prev_label"] = label

    debug_text = _classification_summary(label, confidence, probs, order)
    return "\n\n".join(blocks), debug_text, trace, new_state


def _inject_style() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

          html, body, [class*="css"]  { font-family: 'IBM Plex Sans', sans-serif !important; }

          section[data-testid="stMain"] .block-container,
          .main .block-container {
            max-width: min(92vw, 920px) !important;
            width: 100% !important;
            min-height: auto !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            padding-top: 0.75rem !important;
            padding-bottom: 1.5rem !important;
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
            font-size: clamp(1.65rem, 3.5vw, 2.1rem);
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
            margin: 0 0 0.75rem 0;
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
            margin-bottom: 0.65rem;
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


def _render_scores_expander(label: str, confidence: float, probs: dict[str, float], order: list[str]) -> None:
    with st.expander("Model score currently", expanded=False):
        st.caption(f"Top label: **{label}** ({confidence:.1%})")
        cols = st.columns(len(order))
        for i, name in enumerate(order):
            p = probs.get(name, 0.0)
            with cols[i]:
                st.metric(name.replace("/", " / "), f"{p:.1%}")
        prob_df = pd.DataFrame({"P(class)": [probs.get(k, 0.0) for k in order]}, index=order)
        st.bar_chart(prob_df, use_container_width=True)


def _render_trace_expander(trace: list[str]) -> None:
    with st.expander("Conversation trace", expanded=False):
        st.code("\n".join(trace), language="text")


def main() -> None:
    st.set_page_config(
        page_title="Twinscribe — Mental health detector",
        page_icon="◎",
        layout="centered",
    )
    _inject_style()
    _init_chat_state()

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

    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(
            f'<span class="status-pill">{device.type.upper()} · {" · ".join(order)}</span>',
            unsafe_allow_html=True,
        )
    with h2:
        if st.button("Clear chat", use_container_width=True):
            _clear_chat()
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            shown_content = msg.get("reply") or msg.get("content") or ""
            st.markdown(shown_content)
            if msg["role"] == "assistant" and msg.get("probs") is not None:
                _render_scores_expander(
                    msg.get("label", ""),
                    float(msg.get("confidence", 0.0)),
                    msg["probs"],
                    order,
                )
            if msg["role"] == "assistant" and msg.get("model_details"):
                with st.expander("Model/debug summary", expanded=False):
                    st.markdown(msg["model_details"])
            if msg["role"] == "assistant" and msg.get("trace"):
                _render_trace_expander(msg["trace"])

    if prompt := st.chat_input("Message Twinscribe…"):
        text = prompt.strip()
        if not text:
            st.warning("Empty message after trimming — type something and send again.")
        else:
            st.session_state.messages.append({"role": "user", "content": text})
            label, confidence, probs = run_inference(
                text, model, tokenizer, device, id2label, max_length=int(max_length)
            )
            if not label:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I didn’t get any text to work with after cleaning. Try sending your message again.",
                        "label": None,
                        "confidence": 0.0,
                        "probs": None,
                        "trace": ["engine:contextual_chat", "fallback:nlu_empty"],
                    }
                )
            else:
                reply_text, debug_text, trace, new_state = _build_chatbot_reply(
                    text,
                    label,
                    confidence,
                    probs,
                    order,
                    st.session_state.messages,
                    st.session_state.chatbot_state,
                )
                st.session_state.chatbot_state = new_state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply_text,
                        "reply": reply_text,
                        "model_details": debug_text,
                        "label": label,
                        "confidence": confidence,
                        "probs": probs,
                        "trace": trace,
                    }
                )
            st.rerun()

    st.caption(
        "Crisis: use emergency services or a helpline. Trained on public text; errors happen."
    )


if __name__ == "__main__":
    main()
