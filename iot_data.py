"""IoT data exactly as in the paper (Tables I-II) + a disclosed synthetic image<->sensor pairing."""
import numpy as np
from config import SEED

FEATURES = ["temperature", "humidity", "soil_moisture", "soil_ph", "light_intensity"]
RANGES = {"temperature": (20, 40), "humidity": (30, 90), "soil_moisture": (10, 60),
          "soil_ph": (5.5, 7.5), "light_intensity": (200, 1000)}
T_STEPS = 10


def scale(arr):
    """Min-max scale last axis (5 features) using the fixed Table II ranges."""
    lo = np.array([RANGES[f][0] for f in FEATURES], np.float32)
    hi = np.array([RANGES[f][1] for f in FEATURES], np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


def make_iot_dataset(n=1000, T=T_STEPS, seed=SEED):
    """1000 uniform records; Healthy iff soil_moisture>30 AND humidity>50. Sliding window T, stride 1,
    label = label of the last record. Chronological 80/20 split (no shuffle)."""
    rng = np.random.default_rng(seed)
    rec = np.stack([rng.uniform(*RANGES[f], n) for f in FEATURES], axis=1)
    healthy = ((rec[:, 2] > 30) & (rec[:, 1] > 50)).astype(np.int32)
    X = np.stack([rec[i:i + T] for i in range(n - T + 1)])
    y = healthy[T - 1:]
    X = scale(X)
    cut = int(0.8 * len(X))
    return X[:cut], X[cut:], y[:cut], y[cut:]


_PROFILES = {   # my assumptions (NOT from the paper): typical conditions per class
    "health": dict(temperature=(20, 32), humidity=(45, 70), soil_moisture=(30, 50), soil_ph=(6.0, 7.2), light_intensity=(500, 900)),
    "rust":   dict(temperature=(15, 26), humidity=(65, 90), soil_moisture=(30, 60), soil_ph=(5.5, 7.5), light_intensity=(200, 700)),
}


def make_paired_windows(labels, class_names, T=T_STEPS, noise_frac=0.25, seed=SEED):
    """SYNTHETIC pairing: one sensor window per image, drawn from class-typical conditions.
    noise_frac of samples use uninformative (full-range) conditions so the task is not trivial.
    No real image<->sensor pairs exist, so any fusion gain here reflects these assumptions."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(labels), T, len(FEATURES)), np.float32)
    for i, lab in enumerate(labels):
        name = class_names[int(lab)].lower()
        prof = next((v for k, v in _PROFILES.items() if name.startswith(k)), None)
        if prof is None or rng.random() < noise_frac:
            prof = RANGES
        base = np.array([rng.uniform(*prof[f]) for f in FEATURES])
        walk = np.cumsum(rng.normal(0, 0.01, (T, len(FEATURES))), axis=0) * (np.array([RANGES[f][1] - RANGES[f][0] for f in FEATURES]))
        out[i] = scale(base + walk)
    return out


def window_from_history(df, T=T_STEPS):
    """Last T rows of a DataFrame -> (T, 5) scaled window for the LSTM (missing light -> 600 lux)."""
    d = df.copy()
    if "light_intensity" not in d:
        d["light_intensity"] = 600.0
    arr = d[FEATURES].tail(T).to_numpy(dtype=np.float32)
    if len(arr) < T:
        return None
    return scale(arr)
