"""Train a custom pose classifier with a held-out recording/participant split."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fitness_core import CLASSES, FEATURE_COLUMNS, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent


def split_groups(data, group_column, test_size, seed):
    """Search only for class coverage, never optimize a split using test accuracy."""
    labels = set(data.label)
    if data[group_column].nunique() < 2:
        raise ValueError("Need >=2 independent groups; collect more sessions/participants")
    for train, test in GroupShuffleSplit(n_splits=200, test_size=test_size, random_state=seed).split(
        data, data.label, groups=data[group_column]
    ):
        if set(data.iloc[train].label) == labels and set(data.iloc[test].label) == labels:
            return train, test
    raise ValueError("Cannot place every class in both splits. Record EVERY class in multiple independent groups.")


def train(args):
    data = pd.read_csv(args.data)
    required = FEATURE_COLUMNS + ["label", "schema_version", args.group_by]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("Dataset is empty or contains missing values")
    if not (data.schema_version == SCHEMA_VERSION).all():
        raise ValueError("Dataset schema mismatch")
    if not set(data.label) <= set(CLASSES) or data.label.nunique() < 2:
        raise ValueError("Need at least 2 recognized classes")
    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features contain infinity/NaN")
    if (data.label.value_counts() < 10).any():
        raise ValueError("Collect at least 10 samples per class (100+ across multiple sessions recommended)")
    absent = sorted(set(CLASSES) - set(data.label))
    if absent:
        print("WARNING: model cannot support all exercises. Missing classes:", absent)
    train_idx, test_idx = split_groups(data, args.group_by, args.test_size, args.seed)
    classifier = (RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_leaf=2,
                                        class_weight="balanced", random_state=args.seed, n_jobs=-1)
                  if args.model == "rf" else MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=args.max_iter,
                                                          random_state=args.seed, early_stopping=False))
    pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    pipeline.fit(X.iloc[train_idx], data.label.iloc[train_idx])
    prediction = pipeline.predict(X.iloc[test_idx])
    labels = list(pipeline.classes_)
    accuracy = accuracy_score(data.label.iloc[test_idx], prediction)
    report = classification_report(data.label.iloc[test_idx], prediction, labels=labels, zero_division=0, output_dict=True)
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)} | Grouped by: {args.group_by}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(data.label.iloc[test_idx], prediction, labels=labels, zero_division=0))
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(confusion_matrix(data.label.iloc[test_idx], prediction, labels=labels), annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="GnBu", ax=ax)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"Held-out groups | Accuracy {accuracy:.1%}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    metadata = {"schema_version": SCHEMA_VERSION, "feature_columns": FEATURE_COLUMNS,
                "classes": labels, "trained_at": datetime.now(timezone.utc).isoformat(),
                "sklearn_version": sklearn.__version__, "model_type": args.model,
                "accuracy": accuracy, "report": report, "group_by": args.group_by,
                "train_rows": len(train_idx), "test_rows": len(test_idx),
                "train_groups": sorted(data.iloc[train_idx][args.group_by].astype(str).unique().tolist()),
                "test_groups": sorted(data.iloc[test_idx][args.group_by].astype(str).unique().tolist())}
    joblib.dump({**metadata, "pipeline": pipeline}, args.output / "fitness_model.pkl")
    (args.output / "training_metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output / "fitness_model.pkl")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/pose_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf")
    parser.add_argument("--group-by", choices=["session_id", "subject_id"], default="session_id")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    if not 0 < args.test_size < 1 or args.max_iter < 1:
        parser.error("Invalid test-size or max-iter")
    try:
        train(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
