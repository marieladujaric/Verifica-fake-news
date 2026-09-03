"""Entrena y versiona el modelo desde línea de comandos."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from modeling import build_pipeline, prepare_dataframe, save_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/news.csv")
    parser.add_argument("--output", default="models/fake_news_model.joblib")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base / data_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base / output_path

    raw = pd.read_csv(data_path)
    df, cleaning = prepare_dataframe(raw)
    max_per_class = 4000
    df = pd.concat([
        group.sample(n=min(max_per_class, len(group)), random_state=42)
        for _, group in df.groupby("label")
    ], ignore_index=True)
    cleaning["rows_used_for_modeling"] = len(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    seconds = time.perf_counter() - start
    pred = model.predict(X_test)
    fake_idx = list(model.classes_).index(0)
    p_fake = model.predict_proba(X_test)[:, fake_idx]
    y_fake = (y_test == 0).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_fake": float(precision_score(y_test, pred, pos_label=0)),
        "recall_fake": float(recall_score(y_test, pred, pos_label=0)),
        "f1_fake": float(f1_score(y_test, pred, pos_label=0)),
        "roc_auc_fake": float(roc_auc_score(y_fake, p_fake)),
        "brier_fake": float(brier_score_loss(y_fake, p_fake)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "test_rows": int(len(y_test)),
        "training_seconds": round(seconds, 2),
    }
    metadata = save_bundle(model, output_path, metrics, cleaning)
    report = classification_report(y_test, pred, labels=[0, 1], target_names=["fake", "real"], output_dict=True)
    (output_path.parent / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    run = {
        "python": sys.version,
        "platform": platform.platform(),
        "sklearn": __import__("sklearn").__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "seed": 42,
        "data_path": str(data_path),
        "data_rows": len(df),
        "model_sha256": metadata["sha256"],
    }
    (output_path.parent / "run_manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "model": str(output_path), "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
