"""Funciones compartidas para entrenamiento, inferencia e interpretabilidad."""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

FAKE_LABEL = 0
REAL_LABEL = 1
MODEL_VERSION = "1.0.0"


def clean_text(value: object) -> str:
    """Limpieza conservadora que preserva acentos y señales de estilo."""
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text, flags=re.I)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Límite operativo: reduce memoria/latencia y evita entradas abusivas.
    return text[:3000]


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    work = df[["text", "label"]].copy()
    initial = len(work)
    null_rows = int(work[["text", "label"]].isna().any(axis=1).sum())
    work = work.dropna(subset=["text", "label"])
    work["text"] = work["text"].map(clean_text)
    empty_rows = int(work["text"].str.len().lt(20).sum())
    work = work[work["text"].str.len().ge(20)]
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    invalid_labels = int((~work["label"].isin([0, 1])).sum())
    work = work[work["label"].isin([0, 1])]
    work["label"] = work["label"].astype(int)
    duplicate_rows = int(work.duplicated(subset=["text"]).sum())
    work = work.drop_duplicates(subset=["text"]).reset_index(drop=True)
    report = {
        "rows_initial": initial,
        "rows_final": len(work),
        "null_rows_removed": null_rows,
        "short_rows_removed": empty_rows,
        "invalid_labels_removed": invalid_labels,
        "duplicate_texts_removed": duplicate_rows,
        "class_counts": {str(k): int(v) for k, v in work["label"].value_counts().sort_index().items()},
    }
    return work, report


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.98, max_features=12000,
            sublinear_tf=True, strip_accents="unicode", lowercase=True,
        )),
        ("char", HashingVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), n_features=4096,
            alternate_sign=False, norm="l2", lowercase=True,
        )),
    ])
    classifier = LogisticRegression(
        C=2.0, max_iter=1200, class_weight="balanced", solver="liblinear",
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def fake_probability(model: Pipeline, text: str) -> float:
    cleaned = clean_text(text)
    class_index = list(model.classes_).index(FAKE_LABEL)
    return float(model.predict_proba([cleaned])[0, class_index])


def get_feature_names(model: Pipeline) -> np.ndarray:
    """Nombres para TF-IDF y posiciones auditables para HashingVectorizer."""
    union = model.named_steps["features"]
    names: list[str] = []
    for branch_name, transformer in union.transformer_list:
        if hasattr(transformer, "get_feature_names_out"):
            branch = [f"{branch_name}__{x}" for x in transformer.get_feature_names_out()]
        else:
            branch = [f"{branch_name}__hash_{i}" for i in range(int(transformer.n_features))]
        names.extend(branch)
    return np.asarray(names, dtype=object)


def explain_prediction(model: Pipeline, text: str, top_n: int = 10) -> list[dict]:
    cleaned = clean_text(text)
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    matrix = features.transform([cleaned]).tocsr()
    names = get_feature_names(model)
    # coef_ positivo empuja hacia clase 1 (real); negativo hacia 0 (falsa).
    contributions = matrix.data * classifier.coef_[0, matrix.indices]
    items = []
    for pos in np.argsort(np.abs(contributions))[::-1][:top_n]:
        raw = float(contributions[pos])
        items.append({
            "feature": str(names[matrix.indices[pos]]).replace("word__", "").replace("char__", ""),
            "direction": "real" if raw > 0 else "fake",
            "contribution": abs(raw),
        })
    return items


def save_bundle(model: Pipeline, path: Path, metrics: dict, cleaning_report: dict) -> dict:
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fake_label": FAKE_LABEL,
        "real_label": REAL_LABEL,
        "supported_interface_languages": ["es", "en"],
        "training_language": "primarily English",
        "metrics": metrics,
        "cleaning_report": cleaning_report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path, compress=3)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["sha256"] = digest
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
