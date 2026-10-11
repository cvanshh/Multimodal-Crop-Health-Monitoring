"""Loading, normalising, splitting and batching multispectral TIFF cubes."""
import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from sklearn.model_selection import train_test_split

from config import IMG_SIZE, WL_MIN, WL_MAX, SEED, DEFAULT_CLASS_NAMES


def wavelengths(n_bands: int) -> np.ndarray:
    """Evenly spaced wavelength axis (nm) for n_bands."""
    return np.linspace(WL_MIN, WL_MAX, n_bands)


def read_cube(source, size: int = IMG_SIZE) -> np.ndarray:
    """Read a TIFF (path or raw bytes) -> float32 cube (H, W, B), resized to size x size."""
    if isinstance(source, (bytes, bytearray)):
        with MemoryFile(bytes(source)) as mf, mf.open() as src:
            arr = src.read().astype(np.float32)
    else:
        with rasterio.open(source) as src:
            arr = src.read().astype(np.float32)
    cube = np.transpose(arr, (1, 2, 0))                  # (H, W, B)
    cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
    if size and (cube.shape[0] != size or cube.shape[1] != size):
        interp = cv2.INTER_AREA if cube.shape[0] > size else cv2.INTER_LINEAR
        chunks = []                                   # cv2.resize handles at most 4 channels per call
        for i in range(0, cube.shape[2], 4):
            c = cv2.resize(np.ascontiguousarray(cube[:, :, i:i + 4]), (size, size), interpolation=interp)
            chunks.append(c[:, :, None] if c.ndim == 2 else c)
        cube = np.concatenate(chunks, axis=2)
    return cube.astype(np.float32)


def harmonize_bands(cube: np.ndarray, target: int) -> np.ndarray:
    """Make the cube have `target` bands: drop a constant trailing mask band, else resample by linear interpolation."""
    # an extra trailing band that is constant (e.g. 65535 everywhere) is a mask/padding band: drop it
    while cube.shape[2] > target and float(cube[:, :, -1].std()) < 1e-6:
        cube = cube[:, :, :-1]
    b = cube.shape[2]
    if b == target:
        return cube
    pos = np.linspace(0, b - 1, target)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, b - 1)
    w = (pos - lo).astype(np.float32)
    return (cube[:, :, lo] * (1 - w) + cube[:, :, hi] * w).astype(np.float32)


def normalize_cube(cube: np.ndarray) -> np.ndarray:
    """Robust per-cube scaling to [0, 1] (1st-99th percentile)."""
    lo, hi = np.percentile(cube, [1, 99])
    if hi - lo < 1e-8:
        return np.zeros_like(cube, dtype=np.float32)
    return np.clip((cube - lo) / (hi - lo), 0, 1).astype(np.float32)


def list_classes(data_dir) -> list:
    data_dir = Path(data_dir)
    names = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    return names or DEFAULT_CLASS_NAMES


def load_dataset(data_dir, class_names=None, verbose=True, size=IMG_SIZE, target_bands=None):
    """Load labelled folders -> X (N,H,W,B) float16 normalised, y (N,) int32, class_names, paths.
    Files with a different band count (e.g. 126 vs 125) are resampled to the most common count, not dropped."""
    from collections import Counter

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    class_names = class_names or list_classes(data_dir)
    cubes, y, paths = [], [], []
    for ci, cname in enumerate(class_names):
        cdir = data_dir / cname
        if not cdir.exists():
            print(f"[warn] missing class folder {cdir}")
            continue
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith((".tif", ".tiff")))
        if verbose:
            print(f"Loading {len(files):4d} images from '{cname}'")
        for f in files:
            p = cdir / f
            try:
                cubes.append(read_cube(p, size))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] skipping {p.name}: {e}")
                continue
            y.append(ci)
            paths.append(str(p))
    if not cubes:
        raise RuntimeError(f"No TIFF images found under {data_dir}")
    counts = Counter(c.shape[2] for c in cubes)
    target = target_bands or counts.most_common(1)[0][0]
    if verbose:
        print(f"Band counts found: {dict(counts)} -> using {target} bands")
        n_fix = sum(v for k, v in counts.items() if k != target)
        if n_fix:
            print(f"[info] {n_fix} files had a different band count and were fixed to {target} bands (constant trailing band dropped, otherwise resampled)")
    X = np.stack([normalize_cube(harmonize_bands(c, target)).astype(np.float16) for c in cubes])
    y = np.array(y, dtype=np.int32)
    if verbose:
        print(f"Dataset: X={X.shape}, classes={class_names}, counts={np.bincount(y).tolist()}")
    return X, y, class_names, paths


def split_dataset(X, y, val_size=0.15, test_size=0.15):
    """Stratified train / val / test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)
    rel_val = val_size / (1 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=rel_val, stratify=y_tr, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def make_dataset(X, y, batch_size=16, training=False, S=None):
    """tf.data pipeline. If S (sensor windows) is given, yields ((image, sensors), label)."""
    import tensorflow as tf

    n = len(X)

    def gen():
        idx = np.arange(n)
        if training:
            np.random.shuffle(idx)
        for i in idx:
            x = X[i].astype(np.float32)
            if S is None:
                yield x, np.int32(y[i])
            else:
                yield (x, S[i].astype(np.float32)), np.int32(y[i])

    img_spec = tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32)
    lab_spec = tf.TensorSpec(shape=(), dtype=tf.int32)
    sig = (img_spec, lab_spec) if S is None else (
        (img_spec, tf.TensorSpec(shape=S.shape[1:], dtype=tf.float32)), lab_spec)
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

    if training:
        def aug_img(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            x = x * tf.random.uniform([], 0.9, 1.1)          # brightness
            return tf.clip_by_value(x, 0.0, 1.0)

        if S is None:
            ds = ds.map(lambda x, l: (aug_img(x), l), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda xs, l: ((aug_img(xs[0]), xs[1]), l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    nb = math.ceil(n / batch_size)
    try:                                   # tells Keras the epoch length (removes the "ran out of data" warning)
        ds = ds.apply(tf.data.experimental.assert_cardinality(nb))
    except Exception:  # noqa: BLE001
        pass
    return ds.prefetch(tf.data.AUTOTUNE)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default
