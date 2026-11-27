"""Train per-exercise pose classifiers with a held-out recording/participant split."""
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
from fitness_core import CLASSES, EXERCISES, FEATURE_COLUMNS, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent

# Map each exercise key to its class prefixes.
EXERCISE_CLASSES = {ex: [c for c in CLASSES if c.startswith(ex + "_")] for ex in EXERCISES}


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


def train_one_exercise(exercise, data, args):
    """Train a single per-exercise model. Returns metadata dict or None on skip."""
    ex_classes = EXERCISE_CLASSES[exercise]
    subset = data[data.label.isin(ex_classes)].copy()
    if subset.empty:
        print(f"\n--- {exercise} --- SKIPPED: no rows for {ex_classes}")
        return None
    present = sorted(subset.label.unique())
    missing_cls = sorted(set(ex_classes) - set(present))
    if missing_cls:
        print(f"\n--- {exercise} --- SKIPPED: missing classes {missing_cls}")
        return None
    if (subset.label.value_counts() < 10).any():
        low = subset.label.value_counts()
        low_classes = sorted(low[low < 10].index.tolist())
        print(f"\n--- {exercise} --- SKIPPED: <10 samples for {low_classes}")
        return None

    X = subset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        print(f"\n--- {exercise} --- SKIPPED: features contain infinity/NaN")
        return None

    # Try GroupShuffleSplit; fall back to random 75/25 if only 1 group has all classes.
    fallback = False
    try:
        train_idx, test_idx = split_groups(subset.reset_index(drop=True), args.group_by, args.test_size, args.seed)
    except ValueError as exc:
        print(f"\n--- {exercise} --- WARNING: {exc}")
        print(f"    Falling back to random {1 - args.test_size:.0%}/{args.test_size:.0%} split (no group isolation).")
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(subset))
        split_at = max(1, int(len(subset) * (1 - args.test_size)))
        train_idx, test_idx = perm[:split_at], perm[split_at:]
        fallback = True

    subset = subset.reset_index(drop=True)
    classifier = (RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_leaf=2,
                                        class_weight="balanced", random_state=args.seed, n_jobs=-1)
                  if args.model == "rf" else MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=args.max_iter,
                                                          random_state=args.seed, early_stopping=False))
    pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    pipeline.fit(X.iloc[train_idx], subset.label.iloc[train_idx])
    prediction = pipeline.predict(X.iloc[test_idx])
    labels = list(pipeline.classes_)
    accuracy = accuracy_score(subset.label.iloc[test_idx], prediction)
    report = classification_report(subset.label.iloc[test_idx], prediction, labels=labels, zero_division=0, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  {exercise.upper()} MODEL  ({len(labels)} classes: {labels})")
    print(f"{'='*60}")
    print(f"  Train: {len(train_idx)} | Test: {len(test_idx)} | Grouped by: {args.group_by}" +
          (" (FALLBACK: random split)" if fallback else ""))
    print(f"  Accuracy: {accuracy:.4f}")
    print(classification_report(subset.label.iloc[test_idx], prediction, labels=labels, zero_division=0))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(subset.label.iloc[test_idx], prediction, labels=labels), annot=True, fmt="d",
                xticklabels=labels, yticklabels=labels, cmap="GnBu", ax=ax)
    ax.set(xlabel="Predicted class", ylabel="True class", title=f"{exercise.upper()} | Accuracy {accuracy:.1%}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.output / f"{exercise}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    metadata = {"schema_version": SCHEMA_VERSION, "feature_columns": FEATURE_COLUMNS,
                "classes": labels, "trained_at": datetime.now(timezone.utc).isoformat(),
                "sklearn_version": sklearn.__version__, "model_type": args.model,
                "accuracy": accuracy, "report": report, "group_by": args.group_by,
                "exercise": exercise, "fallback_split": fallback,
                "train_rows": len(train_idx), "test_rows": len(test_idx),
                "train_groups": sorted(subset.iloc[train_idx][args.group_by].astype(str).unique().tolist()) if not fallback else [],
                "test_groups": sorted(subset.iloc[test_idx][args.group_by].astype(str).unique().tolist()) if not fallback else []}
    model_path = args.output / f"{exercise}_model.pkl"
    joblib.dump({**metadata, "pipeline": pipeline}, model_path)
    (args.output / f"{exercise}_training_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {model_path}")
    return metadata


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

    args.output.mkdir(parents=True, exist_ok=True)

    exercises = [args.exercise] if args.exercise != "all" else list(EXERCISES.keys())
    results = {}
    for ex in exercises:
        meta = train_one_exercise(ex, data, args)
        if meta is not None:
            results[ex] = meta

    if not results:
        raise ValueError("No exercise model was trained successfully. Check your dataset.")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(results)}/{len(exercises)} models trained")
    print(f"{'='*60}")
    for ex, meta in results.items():
        print(f"  {ex:>8}: {meta['accuracy']:.1%} accuracy, {len(meta['classes'])} classes")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/pose_data.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf")
    parser.add_argument("--exercise", choices=list(EXERCISES.keys()) + ["all"], default="all",
                        help="Train model for one exercise or all (default: all)")
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
