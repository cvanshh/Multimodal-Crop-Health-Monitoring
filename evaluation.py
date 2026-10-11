"""Metrics + plots for the CNN (3-class), LSTM (binary) and fusion models."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

from config import MODEL_DIR, CNN_PATH, CLASSES_PATH, TEST_DIR, META_PATH
from data_pipeline import load_json, load_dataset


def predict_batches(model, inputs, batch=32):
    multi = isinstance(inputs, (list, tuple))
    n = len(inputs[0]) if multi else len(inputs)
    out = []
    for i in range(0, n, batch):
        b = [a[i:i + batch].astype(np.float32) for a in inputs] if multi else inputs[i:i + batch].astype(np.float32)
        out.append(model.predict(b, verbose=0))
    return np.concatenate(out)


def plot_history(history: dict, out_dir=MODEL_DIR, tag="cnn"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in ("accuracy", "val_accuracy"):
        if k in history: ax[0].plot(history[k], label=k)
    ax[0].set_title(f"{tag} accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    for k in ("loss", "val_loss"):
        if k in history: ax[1].plot(history[k], label=k)
    ax[1].set_title(f"{tag} loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(Path(out_dir) / f"{tag}_training_history.png", dpi=140); plt.close(fig)


def _cm_plot(cm, names, path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def evaluate_multiclass(probs, y, class_names, out_dir=MODEL_DIR, tag="cnn"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred = probs.argmax(1)
    labels = list(range(len(class_names)))
    rep = classification_report(y, pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    print(f"\n[{tag}]\n" + classification_report(y, pred, labels=labels, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    _cm_plot(cm, class_names, out_dir / f"{tag}_confusion_matrix.png", f"{tag} confusion matrix")
    roc = {}
    yb = label_binarize(y, classes=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, c in enumerate(class_names):
        if yb[:, i].sum() == 0 or yb[:, i].sum() == len(y): continue
        fpr, tpr, _ = roc_curve(yb[:, i], probs[:, i]); roc[c] = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc[c]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4); ax.set_title(f"{tag} ROC"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_roc_curves.png", dpi=140); plt.close(fig)
    m = {"accuracy": float(accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "macro_auc": float(np.mean(list(roc.values()))) if roc else None,
         "per_class": {c: rep[c] for c in class_names}, "roc_auc": roc,
         "confusion_matrix": cm.tolist(), "n_samples": int(len(y))}
    print(f"{tag}: accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  macro-AUC={m['macro_auc']}")
    return m


def evaluate_binary(p_healthy, y, out_dir=MODEL_DIR, tag="lstm", names=("Unhealthy", "Healthy")):
    p = np.asarray(p_healthy).ravel(); pred = (p > 0.5).astype(int)
    rep = classification_report(y, pred, labels=[0, 1], target_names=list(names), output_dict=True, zero_division=0)
    print(f"\n[{tag}]\n" + classification_report(y, pred, labels=[0, 1], target_names=list(names), zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    _cm_plot(cm, list(names), Path(out_dir) / f"{tag}_confusion_matrix.png", f"{tag} confusion matrix")
    m = {"accuracy": float(accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "macro_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
         "per_class": {c: rep[c] for c in names}, "confusion_matrix": cm.tolist(), "n_samples": int(len(y))}
    print(f"{tag}: accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  AUC={m['macro_auc']}")
    return m


if __name__ == "__main__":
    from tensorflow import keras
    ap = argparse.ArgumentParser(description="Evaluate the trained CNN on a labelled folder")
    ap.add_argument("--data_dir", default=str(TEST_DIR))
    a = ap.parse_args()
    meta = load_json(META_PATH, {})
    mdl = keras.models.load_model(CNN_PATH, compile=False)
    names = load_json(CLASSES_PATH, None)
    X, y, names, _ = load_dataset(a.data_dir, class_names=names, size=meta.get("img_size", 128))
    evaluate_multiclass(predict_batches(mdl, X), y, names, tag="cnn_external")
