"""Train everything described in the paper and report MEASURED results next to the paper's.

    python train.py --epochs 40                (CNN + LSTM + fusion)
    python train.py --epochs 40 --baselines    (+ SVM and RGB-CNN baselines)
    python train.py --img_size 64              (skip the 64->128 upsampling)
"""
import argparse
import math

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from config import (DATA_DIR, MODEL_DIR, CNN_PATH, LSTM_PATH, FUSION_PATH, CLASSES_PATH, HISTORY_PATH,
                    REFSPEC_PATH, META_PATH, RESULTS_PATH, SEED, IMG_SIZE)
from data_pipeline import load_dataset, split_dataset, make_dataset, save_json, wavelengths
from indices import band_idx
from iot_data import make_iot_dataset, make_paired_windows, FEATURES, T_STEPS
from model import build_cnn, build_lstm, build_fusion
from evaluation import predict_batches, evaluate_multiclass, evaluate_binary, plot_history

PAPER = {  # values REPORTED in the paper (fusion was "projected", not measured; SVM/RGB from other works)
    "cnn": {"accuracy": 0.8760, "macro_f1": 0.8752, "macro_auc": 0.9584},
    "lstm": {"accuracy": 0.9050, "macro_f1": 0.8986, "macro_auc": 0.9512},
    "fusion": {"accuracy": 0.9340, "macro_f1": 0.931, "macro_auc": 0.971},
    "svm": {"accuracy": 0.8900, "macro_f1": 0.881, "macro_auc": None},
    "rgb_cnn": {"accuracy": 0.8430, "macro_f1": 0.836, "macro_auc": None},
}


def callbacks(path=None, patience=10):
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, verbose=1),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=0)]
    if path:
        cb.append(keras.callbacks.ModelCheckpoint(str(path), monitor="val_loss", save_best_only=True, verbose=0))
    return cb


def cw(y):
    u = np.unique(y)
    return {int(c): float(w) for c, w in zip(u, compute_class_weight("balanced", classes=u, y=y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DATA_DIR))
    ap.add_argument("--img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iot_epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    res = {"paper": PAPER, "notes": {}}

    # ------------------------------------------------------------ 1) hyperspectral CNN
    X, y, names, _ = load_dataset(a.data_dir, size=a.img_size)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(X, y, val_size=0.10, test_size=0.20)   # 70/10/20
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}  cube={X.shape[1:]}")
    save_json(names, CLASSES_PATH)
    save_json({"img_size": a.img_size, "num_bands": int(X.shape[3]), "class_names": names,
               "iot_features": FEATURES, "timesteps": T_STEPS}, META_PATH)
    spec = X_tr.mean(axis=(1, 2), dtype=np.float32)
    save_json({"wavelengths": wavelengths(X.shape[3]).tolist(),
               "classes": {c: {"mean": spec[y_tr == i].mean(0).tolist(), "std": spec[y_tr == i].std(0).tolist()}
                           for i, c in enumerate(names) if (y_tr == i).any()}}, REFSPEC_PATH)

    cnn = build_cnn(X.shape[1:], len(names))
    h = cnn.fit(make_dataset(X_tr, y_tr, a.batch_size, True), validation_data=make_dataset(X_va, y_va, a.batch_size),
                epochs=a.epochs, class_weight=cw(y_tr), callbacks=callbacks(CNN_PATH))
    save_json({k: [float(v) for v in vs] for k, vs in h.history.items()}, HISTORY_PATH)
    plot_history(h.history, MODEL_DIR, "cnn")
    cnn = keras.models.load_model(CNN_PATH, compile=False)
    cnn.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    res["cnn"] = evaluate_multiclass(predict_batches(cnn, X_te), y_te, names, tag="cnn")

    # ------------------------------------------------------------ 2) IoT LSTM (simulated data, as in the paper)
    Xs_tr, Xs_te, ys_tr, ys_te = make_iot_dataset()
    cut = int(0.9 * len(Xs_tr))
    lstm = build_lstm(T_STEPS, len(FEATURES))
    h2 = lstm.fit(Xs_tr[:cut], ys_tr[:cut], validation_data=(Xs_tr[cut:], ys_tr[cut:]), epochs=a.iot_epochs,
                  batch_size=32, callbacks=callbacks(LSTM_PATH, 8), verbose=2)
    plot_history(h2.history, MODEL_DIR, "lstm")
    lstm = keras.models.load_model(LSTM_PATH, compile=False)
    lstm.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    res["lstm"] = evaluate_binary(lstm.predict(Xs_te, verbose=0), ys_te, MODEL_DIR, "lstm")
    res["notes"]["lstm"] = ("Simulated uniform sensor data; label is a fixed rule on the LAST record "
                            "(moisture>30 & humidity>50), so this benchmark is nearly trivial.")

    # ------------------------------------------------------------ 3) late fusion (MEASURED, synthetic pairing)
    S_tr = make_paired_windows(y_tr, names, seed=SEED)
    S_va = make_paired_windows(y_va, names, seed=SEED + 1)
    S_te = make_paired_windows(y_te, names, seed=SEED + 2)
    fusion = build_fusion(cnn, lstm, len(names))
    h3 = fusion.fit(make_dataset(X_tr, y_tr, a.batch_size, True, S_tr),
                    validation_data=make_dataset(X_va, y_va, a.batch_size, False, S_va),
                    epochs=a.epochs, class_weight=cw(y_tr), callbacks=callbacks(None, 10))
    plot_history(h3.history, MODEL_DIR, "fusion")
    fusion.save(FUSION_PATH)
    res["fusion"] = evaluate_multiclass(predict_batches(fusion, [X_te, S_te]), y_te, names, tag="fusion")
    res["notes"]["fusion"] = ("Sensor windows are SYNTHETICALLY paired to images using class-typical conditions I "
                              "assumed (25% uninformative). No real image-sensor pairs exist, so the fusion gain "
                              "only demonstrates the architecture, not agronomic value.")

    # ------------------------------------------------------------ 4) optional baselines (same split)
    if a.baselines:
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import SVC
        f = lambda A: np.hstack([A.mean(axis=(1, 2), dtype=np.float32), A.std(axis=(1, 2), dtype=np.float32)])  # noqa: E731
        svm = make_pipeline(StandardScaler(), SVC(C=10, probability=True, class_weight="balanced")).fit(f(X_tr), y_tr)
        res["svm"] = evaluate_multiclass(svm.predict_proba(f(X_te)), y_te, names, tag="svm")
        ids = [band_idx(w, X.shape[3]) for w in (650, 550, 470)]            # approx R, G, B bands
        rgb = build_cnn((a.img_size, a.img_size, 3), len(names))
        rgb.fit(make_dataset(X_tr[..., ids], y_tr, a.batch_size, True),
                validation_data=make_dataset(X_va[..., ids], y_va, a.batch_size),
                epochs=a.epochs, class_weight=cw(y_tr), callbacks=callbacks(None, 10), verbose=2)
        res["rgb_cnn"] = evaluate_multiclass(predict_batches(rgb, X_te[..., ids]), y_te, names, tag="rgb_cnn")
        res["notes"]["baselines"] = "SVM = RBF-SVM on per-band mean/std; RGB-CNN = same CNN on 3 bands. 3D-CNN not implemented."

    save_json(res, RESULTS_PATH)

    # ------------------------------------------------------------ summary table
    print("\n" + "=" * 74)
    print(f"{'Model':<22}{'Paper acc':>11}{'Measured acc':>14}{'Paper F1':>10}{'Measured F1':>13}")
    print("-" * 74)
    for k, label in [("svm", "SVM (HSI)"), ("rgb_cnn", "RGB-CNN"), ("cnn", "CNN (HSI)"),
                     ("lstm", "LSTM (IoT)"), ("fusion", "Fusion CNN+LSTM")]:
        if k in res:
            p, m = PAPER[k], res[k]
            print(f"{label:<22}{p['accuracy']*100:>10.1f}%{m['accuracy']*100:>13.1f}%{p['macro_f1']:>10.3f}{m['macro_f1']:>13.3f}")
    print("=" * 74)
    print(f"Test images: {len(y_te)}.  Results saved to {RESULTS_PATH}\n"
          "Fusion uses SYNTHETIC image-sensor pairing - disclose this wherever you report it.")


if __name__ == "__main__":
    main()
