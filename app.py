from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from modeling import clean_text, explain_prediction, fake_probability

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "fake_news_model.joblib"
LOG_PATH = BASE / "monitoring" / "predictions.jsonl"

st.set_page_config(page_title="VerificaIA | Fake News", page_icon="🔎", layout="wide")

COPY = {
    "Español": {
        "title": "VerificaIA · Detector de noticias falsas",
        "subtitle": "Evaluación probabilística e interpretable para textos en español e inglés",
        "input": "Pega el título y el cuerpo de la noticia",
        "placeholder": "Escribe o pega aquí la noticia completa…",
        "button": "Analizar noticia",
        "fake": "Probabilidad de ser falsa",
        "real": "Probabilidad de ser real",
        "signals": "Señales que influyeron en el resultado",
        "warning": "Este resultado es apoyo a la verificación, no una prueba de verdad. Confirma fuente, autor, fecha y evidencia.",
        "short": "Incluye al menos 80 caracteres para obtener una evaluación más estable.",
        "scope": "El modelo fue entrenado principalmente con noticias en inglés. Acepta español mediante rasgos de palabras y caracteres, pero necesita validación con un corpus español etiquetado antes de usos críticos.",
    },
    "English": {
        "title": "VerificaIA · Fake news detector",
        "subtitle": "Interpretable probabilistic assessment for Spanish and English text",
        "input": "Paste the headline and article body",
        "placeholder": "Write or paste the full article here…",
        "button": "Analyze article",
        "fake": "Probability of being fake",
        "real": "Probability of being real",
        "signals": "Signals influencing the result",
        "warning": "This score supports verification; it is not proof of truth. Confirm the source, author, date, and evidence.",
        "short": "Provide at least 80 characters for a more stable assessment.",
        "scope": "The model was trained primarily on English news. It accepts Spanish through word and character features, but must be validated with a labeled Spanish corpus before critical use.",
    },
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def log_prediction(text: str, probability: float, metadata: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata.get("model_version"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "probability_fake": round(probability, 6),
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


language = st.sidebar.radio("Idioma / Language", ["Español", "English"])
t = COPY[language]
st.title(t["title"])
st.caption(t["subtitle"])

if not MODEL_PATH.exists():
    st.error("Modelo no encontrado. Ejecuta primero `python train_model.py`.")
    st.stop()

bundle = load_bundle()
model, metadata = bundle["model"], bundle["metadata"]
with st.sidebar:
    st.markdown("### Model card")
    st.write(f"Version: **{metadata.get('model_version', 'N/A')}**")
    score = metadata.get("metrics", {}).get("f1_fake")
    if score is not None:
        st.write(f"F1 (fake, test): **{score:.1%}**")
    st.info(t["scope"])

text = st.text_area(t["input"], height=260, placeholder=t["placeholder"])
if st.button(t["button"], type="primary", use_container_width=True):
    cleaned = clean_text(text)
    if len(cleaned) < 80:
        st.warning(t["short"])
    else:
        p_fake = fake_probability(model, cleaned)
        p_real = 1.0 - p_fake
        left, right = st.columns(2)
        left.metric(t["fake"], f"{p_fake:.1%}")
        right.metric(t["real"], f"{p_real:.1%}")
        st.progress(p_fake)
        if p_fake >= 0.70:
            st.error("Riesgo alto / High risk")
        elif p_fake >= 0.40:
            st.warning("Resultado incierto / Uncertain result")
        else:
            st.success("Riesgo bajo / Low risk")
        signals = explain_prediction(model, cleaned, top_n=12)
        table = pd.DataFrame(signals)
        if not table.empty:
            table["contribution"] = table["contribution"].round(4)
            st.subheader(t["signals"])
            st.dataframe(table, use_container_width=True, hide_index=True)
        st.info(t["warning"])
        log_prediction(cleaned, p_fake, metadata)

