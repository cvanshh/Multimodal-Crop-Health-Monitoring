"""IoT sensor simulation/ingestion, rule-based alerts, disease-risk score, LSTM forecasting."""
import numpy as np
import pandas as pd

SENSOR_COLS = ["temperature", "soil_moisture", "humidity", "soil_ph"]
UNITS = {"temperature": "°C", "soil_moisture": "%", "humidity": "%", "soil_ph": "pH"}

# (low, high) healthy ranges per crop
CROP_PROFILES = {
    "Wheat":   {"temperature": (12, 28), "soil_moisture": (25, 40), "humidity": (40, 70), "soil_ph": (6.0, 7.5)},
    "Rice":    {"temperature": (22, 34), "soil_moisture": (40, 60), "humidity": (60, 85), "soil_ph": (5.5, 6.8)},
    "Maize":   {"temperature": (18, 32), "soil_moisture": (25, 40), "humidity": (45, 75), "soil_ph": (5.8, 7.0)},
    "Cotton":  {"temperature": (20, 35), "soil_moisture": (20, 35), "humidity": (40, 70), "soil_ph": (5.8, 8.0)},
    "Generic": {"temperature": (15, 32), "soil_moisture": (25, 35), "humidity": (40, 70), "soil_ph": (6.0, 7.0)},
}


def simulate_current(seed=None):
    r = np.random.default_rng(seed)
    return {"temperature": round(float(r.uniform(18, 36)), 1),
            "soil_moisture": round(float(r.uniform(15, 45)), 1),
            "humidity": round(float(r.uniform(40, 85)), 1),
            "soil_ph": round(float(r.uniform(5.5, 7.8)), 1)}


def simulate_history(days=60, seed=7):
    """Seasonal + noise history with irrigation/rain events (stand-in for a real IoT feed)."""
    r = np.random.default_rng(seed)
    t = np.arange(days)
    temp = 26 + 4 * np.sin(2 * np.pi * t / 30) + r.normal(0, 1.2, days)
    hum = 62 + 10 * np.sin(2 * np.pi * t / 20 + 1) + r.normal(0, 3, days)
    moist = np.zeros(days); moist[0] = 32
    for i in range(1, days):
        moist[i] = moist[i - 1] - 0.9 - 0.04 * (temp[i] - 25) + r.normal(0, .5)
        if moist[i] < 22 or r.random() < 0.06:
            moist[i] += r.uniform(8, 14)
    moist = np.clip(moist, 10, 50)
    ph = 6.6 + np.cumsum(r.normal(0, 0.01, days))
    light = np.clip(600 + 250 * np.sin(2 * np.pi * t / 7) + r.normal(0, 60, days), 200, 1000)
    ndvi = np.clip(0.55 + 0.12 * np.sin(2 * np.pi * t / 60) + 0.002 * (moist - 30) + r.normal(0, .015, days), .2, .9)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame({"date": dates, "temperature": temp.round(1), "soil_moisture": moist.round(1),
                         "humidity": hum.round(1), "soil_ph": ph.round(2),
                         "light_intensity": light.round(0), "ndvi": ndvi.round(3)})


def read_sensor_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Needed: date, {', '.join(SENSOR_COLS)} (+ optional ndvi)")
    if "light_intensity" not in df.columns:
        df["light_intensity"] = 600.0
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df))
    return df.sort_values("date").reset_index(drop=True)


def evaluate_sensors(data: dict, crop="Wheat"):
    prof = CROP_PROFILES.get(crop, CROP_PROFILES["Generic"])
    status, alerts, recs = {}, [], []
    msgs = {
        "temperature": ("Cold stress risk", "Heat stress risk",
                        "Protect crops from cold (mulch / cover).", "Irrigate in early morning/evening; consider shade nets."),
        "soil_moisture": ("Water stress detected", "Waterlogging risk",
                          "Irrigate within 12 hours.", "Pause irrigation and check drainage."),
        "humidity": ("Low humidity", "High humidity - fungal disease risk",
                     "Monitor transpiration; light irrigation helps.", "Scout for fungal disease; improve airflow; plan preventive fungicide with an agronomist."),
        "soil_ph": ("Acidic soil", "Alkaline soil",
                    "Apply agricultural lime per soil-test advice.", "Use sulphur / organic matter per soil-test advice."),
    }
    for k in SENSOR_COLS:
        lo, hi = prof[k]; v = data[k]
        if v < lo:
            status[k] = "low"; alerts.append(f"{msgs[k][0]} ({v}{UNITS[k]} < {lo})"); recs.append(msgs[k][2])
        elif v > hi:
            status[k] = "high"; alerts.append(f"{msgs[k][1]} ({v}{UNITS[k]} > {hi})"); recs.append(msgs[k][3])
        else:
            status[k] = "normal"
    return status, alerts, recs


def disease_risk(temp, humidity, moisture):
    """Heuristic 0-100 fungal/rust *weather* risk (rust favours ~15-25 °C and high humidity)."""
    tf = max(0.0, 1 - abs(temp - 20) / 10)
    hf = float(np.clip((humidity - 50) / 40, 0, 1))
    mf = float(np.clip((moisture - 25) / 25, 0, 1))
    score = 100 * (0.40 * tf + 0.45 * hf + 0.15 * mf)
    level = "High" if score >= 65 else "Medium" if score >= 40 else "Low"
    return round(score, 1), level


def health_score(ndvi_mean, stressed_frac, p_healthy):
    """Composite 0-100 field-health score (transparent heuristic, not a calibrated metric)."""
    s = 0.5 * np.clip(ndvi_mean / 0.8, 0, 1) + 0.3 * (1 - stressed_frac) + 0.2 * p_healthy
    return round(float(100 * s), 1)


def image_advice(pred_class, conf, stressed_pct):
    recs = []
    if pred_class.lower().startswith("rust") and conf >= 0.5:
        recs += ["Rust signature detected: scout the highlighted zones on foot within 24-48 h.",
                 "If confirmed, consult your local agri-extension for a registered fungicide and timing."]
    if stressed_pct > 30:
        recs.append(f"{stressed_pct:.0f}% of the area is stressed (NDVI < 0.3): check nutrients, water and pests.")
    if conf < 0.6:
        recs.append("Model confidence is low - re-capture the image or verify in the field.")
    return recs


# ---------------- forecasting ----------------
def linear_forecast(values, horizon=7, window=14):
    v = np.asarray(values, float)[-window:]
    x = np.arange(len(v))
    a, b = np.polyfit(x, v, 1)
    mean = a * (len(v) + np.arange(horizon)) + b
    sd = np.std(v - (a * x + b)) or 1e-6
    band = sd * np.sqrt(1 + np.arange(1, horizon + 1) / 3)
    return mean, mean - 1.64 * band, mean + 1.64 * band


def lstm_forecast(values, horizon=7, window=14, epochs=60):
    """Tiny LSTM trained on the series itself. Falls back to linear if TF/data is unavailable."""
    v = np.asarray(values, float)
    if len(v) < window + horizon + 8:
        return linear_forecast(v, horizon, window)
    try:
        import tensorflow as tf
        from tensorflow import keras
        keras.utils.set_random_seed(42)
        mu, sd = v.mean(), v.std() + 1e-8
        z = (v - mu) / sd
        X = np.array([z[i:i + window] for i in range(len(z) - window - horizon + 1)])[..., None]
        Y = np.array([z[i + window:i + window + horizon] for i in range(len(z) - window - horizon + 1)])
        m = keras.Sequential([keras.Input((window, 1)), keras.layers.LSTM(32), keras.layers.Dense(horizon)])
        m.compile("adam", "mse")
        m.fit(X, Y, epochs=epochs, batch_size=16, verbose=0)
        resid = (m.predict(X, verbose=0) - Y).std(axis=0) * sd
        pred = m.predict(z[-window:][None, :, None], verbose=0)[0] * sd + mu
        band = 1.64 * np.maximum(resid, 1e-6)
        return pred, pred - band, pred + band
    except Exception:  # noqa: BLE001
        return linear_forecast(v, horizon, window)
