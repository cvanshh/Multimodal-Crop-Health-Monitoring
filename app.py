"""AgriSpectra.AI - Streamlit dashboard.   Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import MODEL_DIR
from data_pipeline import read_cube, normalize_cube, harmonize_bands
from indices import (compute_indices, zone_percentages, zone_rgb, ndvi_stats, mean_spectrum,
                     ZONE_NAMES, ZONE_COLORS)
from imaging import true_color, color_infrared, overlay_heatmap, upscale, synthetic_cube
from inference import Predictor, heuristic_probs
from sensors import (CROP_PROFILES, SENSOR_COLS, UNITS, simulate_history, read_sensor_csv,
                     evaluate_sensors, disease_risk, health_score, image_advice,
                     lstm_forecast, linear_forecast)
from report import build_html, build_json, build_csv
from iot_data import window_from_history

st.set_page_config(page_title="AgriSpectra.AI", page_icon="🌾", layout="wide")
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#1b5e20,#2e7d32 55%,#558b2f);padding:22px 28px;border-radius:14px;margin-bottom:14px}
.hero h1{margin:0;color:#fff;font-size:1.9rem}.hero p{margin:4px 0 0;color:#e0f2e1}
[data-testid="stMetric"]{background:#17241c;border:1px solid #25402e;padding:12px 14px;border-radius:12px}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600}
.ok{background:#1b5e20;color:#c8e6c9}.warn{background:#6d4c00;color:#ffe082}.bad{background:#7f1d1d;color:#ffcdd2}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🌾 AgriSpectra.AI</h1>'
            '<p>Hyperspectral crop-health monitoring · CNN + LSTM disease detection · IoT sensors · forecasting</p></div>',
            unsafe_allow_html=True)


# ------------------------------------------------------------------ cached helpers
@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    return Predictor()


@st.cache_data(show_spinner=False)
def load_cube_bytes(data: bytes, size: int, bands=None):
    cube = read_cube(data, size)
    return harmonize_bands(cube, bands) if bands else cube


@st.cache_data(show_spinner="Analysing…")
def analyze(name, cube, _pred, model_sig):
    norm = normalize_cube(cube)
    ind = compute_indices(cube)
    ndvi = ind["NDVI"]
    names = list(_pred.class_names)
    ok = _pred.available
    if ok:
        try:
            probs = _pred.predict(norm)
            mode = "CNN"
        except Exception:  # noqa: BLE001  (e.g. wrong number of bands)
            ok = False
    if ok:
        try:
            cam = _pred.gradcam(norm)
        except Exception:  # noqa: BLE001
            cam = None
    else:
        probs, cam, mode = heuristic_probs(float(ndvi.mean()), names), None, "Heuristic (no trained model)"
    if len(probs) != len(names):
        names = [f"class_{i}" for i in range(len(probs))]
    k = int(np.argmax(probs))
    return dict(name=name, cube=cube, norm=norm, ind=ind, zones=zone_percentages(ndvi),
                stats=ndvi_stats(ndvi), probs=probs, names=names, pred=names[k],
                conf=float(probs[k]), mode=mode, cam=cam)


@st.cache_data(show_spinner="Forecasting…")
def run_forecast(values: tuple, horizon: int, method: str):
    arr = np.array(values, float)
    return (lstm_forecast if method.startswith("LSTM") else linear_forecast)(arr, horizon)


def style(fig, h=320):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def show_rgb(arr, title):
    fig = px.imshow(arr)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(title=title)
    st.plotly_chart(style(fig, 360))


def ndvi_map(R):
    fig = px.imshow(R["ind"]["NDVI"], color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                    title="NDVI map")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    fig.update_layout(coloraxis_colorbar=dict(title="NDVI"))
    return style(fig, 380)


def health_map(R):
    fig = px.imshow(upscale(zone_rgb(R["ind"]["NDVI"]), factor=6, nearest=True), title="Health map")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return style(fig, 380)


def zone_legend():
    rng = ["< 0.1", "0.1–0.3", "0.3–0.5", "0.5–0.7", "> 0.7"]
    return " ".join(f'<span class="pill" style="background:{c};color:#000">{n} ({r})</span>'
                    for n, c, r in zip(ZONE_NAMES, ZONE_COLORS, rng))


def gauge(value, title, band, rng, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix}, title={"text": title},
        gauge={"axis": {"range": list(rng)}, "bar": {"color": "#81c784"},
               "steps": [{"range": [rng[0], band[0]], "color": "#5d4037"},
                         {"range": list(band), "color": "#1b5e20"},
                         {"range": [band[1], rng[1]], "color": "#5d4037"}]}))
    return style(fig, 210)


# ------------------------------------------------------------------ sidebar
pred = get_predictor()

with st.sidebar:
    st.header("⚙️ Control panel")

    source = st.radio(
        "Imagery source",
        ["Upload TIFF(s)", "Demo scenes"]
    )

    crop = st.selectbox(
        "Crop profile",
        list(CROP_PROFILES)
    )

    sensor_src = st.radio(
        "Sensor feed",
        ["Simulated live feed", "Manual input", "Upload CSV"]
    )

    if "seed" not in st.session_state:
        st.session_state.seed = 7

    manual, csv_file = {}, None

    if sensor_src == "Simulated live feed":
        if st.button("🔄 New reading"):
            st.session_state.seed += 1

    elif sensor_src == "Manual input":
        manual = {
            "temperature": st.slider(
                "Temperature (°C)", 0.0, 50.0, 27.0, 0.5
            ),
            "soil_moisture": st.slider(
                "Soil moisture (%)", 0.0, 80.0, 30.0, 0.5
            ),
            "humidity": st.slider(
                "Humidity (%)", 0.0, 100.0, 60.0, 0.5
            ),
            "soil_ph": st.slider(
                "Soil pH", 3.5, 9.5, 6.6, 0.1
            )
        }

    else:
        csv_file = st.file_uploader(
            "Sensor CSV",
            type=["csv"],
            help="Columns: date, temperature, soil_moisture, "
                 "humidity, soil_ph, [ndvi]"
        )

        st.download_button(
            "⬇ Sample CSV",
            simulate_history(45).to_csv(index=False),
            "sample_sensors.csv"
        )

    st.divider()

    if pred.available:
        st.markdown(
            '<span class="pill ok">CNN + LSTM Fusion</span>',
            unsafe_allow_html=True
        )

        # Read the saved fusion evaluation results
        fusion_results = pred.results or {}
        fusion_metrics = fusion_results.get("fusion")

        if fusion_metrics and fusion_metrics.get("accuracy") is not None:
            fusion_accuracy = fusion_metrics["accuracy"] * 100

            st.caption(
                f"CNN + LSTM Fusion Accuracy: "
                f"**{fusion_accuracy:.1f}%**"
            )
        else:
            st.caption(
                "Fusion accuracy unavailable. "
                "Run python train.py to generate evaluation results."
            )

    else:
        st.markdown(
            '<span class="pill warn">No trained model</span>',
            unsafe_allow_html=True
        )

        st.caption(
            "Run `python train.py` to train. Until then, "
            "class predictions come from a simple NDVI heuristic "
            "and are for demo only."
        )

        if pred.error:
            st.error(pred.error)

# ------------------------------------------------------------------ scenes
scenes = {}
if source == "Upload TIFF(s)":
    files = st.file_uploader("Drop multispectral / hyperspectral .tif files", type=["tif", "tiff"],
                             accept_multiple_files=True)
    for f in files or []:
        try:
            scenes[f.name] = load_cube_bytes(f.getvalue(), pred.img_size, pred.num_bands)
        except Exception as e:  # noqa: BLE001
            st.error(f"{f.name}: {e}")
else:
    scenes = {"Demo · healthy field": synthetic_cube("healthy", size=pred.img_size, seed=1),
              "Demo · rust outbreak": synthetic_cube("rust", size=pred.img_size, seed=2),
              "Demo · patchy infection": synthetic_cube("mixed", size=pred.img_size, seed=3)}
    st.caption("Synthetic demo cubes (not real data) so you can explore the UI.")

if not scenes:
    st.info("⬆ Upload one or more `.tif` cubes (or switch to **Demo scenes** in the sidebar).")
    st.stop()

sig = str(pred.model.count_params()) if pred.available else "none"
results = {n: analyze(n, c, pred, sig) for n, c in scenes.items()}

# ------------------------------------------------------------------ sensors
if sensor_src == "Upload CSV" and csv_file is not None:
    try:
        hist = read_sensor_csv(csv_file)
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(str(e)); hist = simulate_history(60, st.session_state.seed)
else:
    hist = simulate_history(60, st.session_state.seed)
    if sensor_src == "Manual input":
        for k, v in manual.items():
            hist.loc[hist.index[-1], k] = v
current = {k: float(hist[k].iloc[-1]) for k in SENSOR_COLS}
status, s_alerts, s_recs = evaluate_sensors(current, crop)
risk, risk_level = disease_risk(current["temperature"], current["humidity"], current["soil_moisture"])

hist_w = window_from_history(hist)
p_sensor = pred.sensor_status(hist_w)

# ------------------------------------------------------------------ batch table + selection
if len(results) > 1:
    rows = [{"File": n, "Diagnosis": r["pred"], "Confidence %": round(r["conf"] * 100, 1),
             "Mean NDVI": round(r["stats"]["mean"], 3),
             "Stressed %": round(r["zones"][ZONE_NAMES[0]] + r["zones"][ZONE_NAMES[1]], 1)}
            for n, r in results.items()]
    with st.expander(f"📦 Batch summary ({len(results)} images)", expanded=False):
        st.dataframe(pd.DataFrame(rows))
    chosen = st.selectbox("Inspect image", list(results))
else:
    chosen = next(iter(results))
R = results[chosen]

stressed_pct = R["zones"][ZONE_NAMES[0]] + R["zones"][ZONE_NAMES[1]]
p_healthy = float(next((p for c, p in zip(R["names"], R["probs"]) if c.lower().startswith("health")), 0.5))
score = health_score(R["stats"]["mean"], stressed_pct / 100, p_healthy)
is_rust = "rust" in R["pred"].lower()

alerts = list(s_alerts)
if is_rust and R["conf"] >= 0.5:
    alerts.insert(0, f"AI detected rust signature ({R['conf']*100:.0f}% confidence)")
if risk_level == "High":
    alerts.append(f"High fungal-disease weather risk ({risk}/100)")
recs = image_advice(R["pred"], R["conf"], stressed_pct) + s_recs
if risk_level == "High":
    recs.append("Weather favours rust spread: increase scouting frequency this week.")

# ------------------------------------------------------------------ KPIs
c = st.columns(6)
c[0].metric("Field health", f"{score:.0f}/100")
c[1].metric("AI diagnosis", R["pred"])
c[2].metric("Confidence", f"{R['conf']*100:.1f}%")
c[3].metric("Mean NDVI", f"{R['stats']['mean']:.2f}")
c[4].metric("Stressed area", f"{stressed_pct:.1f}%")
c[5].metric("Disease-weather risk", f"{risk_level} ({risk:.0f})")
if R["mode"] != "CNN":
    st.warning("Diagnosis mode: **heuristic demo** (no trained model found). Train the model for real predictions.")

tabs = st.tabs(["🏠 Overview", "🛰️ Imagery", "🌈 Spectral", "🌡️ Sensors", "📈 Forecast", "🧠 Model", "📋 Report"])

# ---- Overview
with tabs[0]:
    a, b = st.columns([1, 1])
    with a:
        st.plotly_chart(gauge(score, "Field health score", (60, 100), (0, 100)))
        pf = px.bar(x=R["probs"] * 100, y=R["names"], orientation="h",
                    labels={"x": "Probability %", "y": ""}, title="Class probabilities")
        pf.update_traces(marker_color="#66bb6a")
        st.plotly_chart(style(pf, 220))
        zf = go.Figure()
        for n, col in zip(ZONE_NAMES, ZONE_COLORS):
            zf.add_trace(go.Bar(y=["Area"], x=[R["zones"][n]], name=n, orientation="h", marker_color=col))
        zf.update_layout(barmode="stack", title="Health-zone distribution (% area)")
        st.plotly_chart(style(zf, 200))
    with b:
        st.subheader("🚨 Detected issues")
        if alerts:
            for t in alerts: st.error(t, icon="⚠️")
        else:
            st.success("No issues detected - crops look healthy.")
        st.subheader("✅ Recommended actions")
        if recs:
            for t in dict.fromkeys(recs): st.info(t, icon="💡")
        else:
            st.success("Continue current practices.")
        st.caption("Advice is rule-based decision support - confirm in the field with an agronomist.")
    m1, m2 = st.columns(2)
    m1.plotly_chart(ndvi_map(R), key="ov_ndvi")
    m2.plotly_chart(health_map(R), key="ov_health")
    m2.markdown(zone_legend(), unsafe_allow_html=True)

# ---- Imagery
with tabs[1]:
    cube = R["cube"]
    c1, c2 = st.columns(2)
    c1.plotly_chart(ndvi_map(R), key="im_ndvi")
    c2.plotly_chart(health_map(R), key="im_health")
    c2.markdown(zone_legend(), unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        show_rgb(upscale(true_color(cube)), "True colour (≈650/550/470 nm)")
    with c4:
        show_rgb(upscale(color_infrared(cube)), "Colour-infrared (NIR/Red/Green) - healthy plants glow red")
    idx_name = st.selectbox("More vegetation indices", list(R["ind"]), index=1)
    m = R["ind"][idx_name]
    fig = px.imshow(m, color_continuous_scale="RdYlGn", zmin=float(np.percentile(m, 2)),
                    zmax=float(np.percentile(m, 98)), title=f"{idx_name} map")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    st.plotly_chart(style(fig, 380))
    st.subheader("Disease attention map (Grad-CAM)")
    if R["cam"] is not None:
        base = upscale(true_color(cube))
        show_rgb(overlay_heatmap(base, R["cam"]), f"Regions driving the '{R['pred']}' decision")
        st.caption("Warm colours = pixels that most influenced the CNN branch. Indicates where to scout, not exact lesion borders.")
    else:
        st.info("Grad-CAM needs a trained model. Train with `python train.py`.")

# ---- Spectral
with tabs[2]:
    view = st.radio("Spectrum view", ["Reflectance", "Normalised (compare with training classes)"], horizontal=True)
    sf = go.Figure()
    if view == "Reflectance":
        wl, mu, sd = mean_spectrum(R["cube"])
        sf.add_trace(go.Scatter(x=wl, y=mu + sd, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        sf.add_trace(go.Scatter(x=wl, y=mu - sd, fill="tonexty", line=dict(width=0),
                                fillcolor="rgba(129,199,132,.25)", name="±1 std"))
        sf.add_trace(go.Scatter(x=wl, y=mu, name=chosen, line=dict(color="#a5d6a7", width=3)))
    else:
        wl = np.linspace(450, 950, R["norm"].shape[2])
        sf.add_trace(go.Scatter(x=wl, y=R["norm"].mean(axis=(0, 1)), name=chosen,
                                line=dict(color="#fff", width=3)))
        ref = pred.reference
        if ref:
            for cname, d in ref["classes"].items():
                sf.add_trace(go.Scatter(x=ref["wavelengths"], y=d["mean"], name=f"train · {cname}",
                                        line=dict(dash="dot")))
        else:
            st.info("Train the model to generate class reference spectra.")
    sf.add_vrect(x0=680, x1=750, fillcolor="orange", opacity=.12, line_width=0,
                 annotation_text="Red edge", annotation_position="top left")
    sf.update_layout(title="Mean spectral signature", xaxis_title="Wavelength (nm)", yaxis_title="Value")
    st.plotly_chart(style(sf, 420))
    st.caption("Stress (e.g. rust) typically raises red reflectance, lowers NIR and shifts/flattens the red edge. "
               "Wavelength axis assumes 450-950 nm - edit WL_MIN/WL_MAX in config.py if your sensor differs.")
    st.dataframe(pd.DataFrame({k: {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                                   "std": float(v.std())} for k, v in R["ind"].items()}).T.round(3))

# ---- Sensors
with tabs[3]:
    prof = CROP_PROFILES[crop]
    ranges = {"temperature": (0, 50), "soil_moisture": (0, 80), "humidity": (0, 100), "soil_ph": (3.5, 9.5)}
    g = st.columns(4)
    for col, k in zip(g, SENSOR_COLS):
        with col:
            st.plotly_chart(gauge(current[k], k.replace("_", " ").title(), prof[k], ranges[k], UNITS[k] if k != "soil_ph" else ""))
            cls = {"normal": "ok", "low": "warn", "high": "bad"}[status[k]]
            st.markdown(f'<span class="pill {cls}">{status[k].upper()}</span> '
                        f'<small>target {prof[k][0]}–{prof[k][1]}</small>', unsafe_allow_html=True)
    if p_sensor is not None:
        lab = "Healthy" if p_sensor > 0.5 else "Unhealthy"
        st.metric("LSTM crop-status (last 10 readings)", f"{lab}", f"P(healthy) = {p_sensor:.2f}", delta_color="off")
        st.caption("The LSTM was trained on simulated data with a fixed rule (moisture > 30 & humidity > 50). "
                   "It reproduces that rule; it is not a validated agronomic model.")
    cols = [c_ for c_ in SENSOR_COLS + (["ndvi"] if "ndvi" in hist else []) if c_ in hist]
    pick = st.multiselect("History", cols, default=["soil_moisture", "temperature"])
    if pick:
        lf = px.line(hist, x="date", y=pick, title="Sensor history")
        st.plotly_chart(style(lf, 360))
    st.caption("Feed: " + {"Simulated live feed": "simulated (demo)", "Manual input": "manual sliders",
                           "Upload CSV": "your CSV"}[sensor_src] + ". Connect real IoT by replacing `simulate_history()` in sensors.py.")

# ---- Forecast
with tabs[4]:
    fc = st.columns(3)
    var = fc[0].selectbox("Variable", [c_ for c_ in SENSOR_COLS + ["ndvi"] if c_ in hist])
    horizon = fc[1].slider("Days ahead", 3, 14, 7)
    method = fc[2].radio("Model", ["LSTM (neural)", "Linear (fast)"], horizontal=True)
    mean, lo, hi = run_forecast(tuple(hist[var].astype(float).tolist()), horizon, method)
    fut = pd.date_range(hist["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    tail = hist.tail(30)
    ff = go.Figure()
    ff.add_trace(go.Scatter(x=tail["date"], y=tail[var], name="history", line=dict(color="#a5d6a7")))
    ff.add_trace(go.Scatter(x=fut, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    ff.add_trace(go.Scatter(x=fut, y=lo, fill="tonexty", line=dict(width=0),
                            fillcolor="rgba(255,213,79,.2)", name="90% band"))
    ff.add_trace(go.Scatter(x=fut, y=mean, name="forecast", line=dict(color="#ffd54f", dash="dash", width=3)))
    ff.update_layout(title=f"{var.replace('_',' ').title()} - next {horizon} days")
    st.plotly_chart(style(ff, 400))
    if var in prof:
        lo_t, hi_t = prof[var]
        below = np.where(mean < lo_t)[0]; above = np.where(mean > hi_t)[0]
        if len(below): st.warning(f"Forecast drops below {lo_t} in ~{below[0]+1} day(s) - plan ahead.")
        elif len(above): st.warning(f"Forecast exceeds {hi_t} in ~{above[0]+1} day(s) - plan ahead.")
        else: st.success("Forecast stays within the target range.")
    st.caption("The LSTM is trained on the selected series each time (needs ≳30 days of data). With real sensor logs, accuracy improves.")

# ---- Model
# ---- Model
with tabs[5]:
    st.markdown(
        "**Architecture**\n"
        "- **CNN branch:** 3 × [Conv3×3 → ReLU → BatchNorm → MaxPool] "
        "(32/64/128) → Dense(256) → Dropout(0.5). Grad-CAM at the last conv.\n"
        "- **LSTM branch:** input 10×5 sensors → LSTM(64) → "
        "Dropout(0.2) → LSTM(32).\n"
        "- **Late fusion:** concat(256+32 = 288) → Dense(64) → "
        "Dropout(0.3) → softmax(3)."
    )

    rs = pred.results

    if rs:
        # Separate CNN + LSTM fusion accuracy
        if "fusion" in rs:
            fusion_accuracy = rs["fusion"]["accuracy"] * 100
            fusion_f1 = rs["fusion"]["macro_f1"]

            col1, col2 = st.columns(2)

            col1.metric(
                label="CNN + LSTM Fusion Accuracy",
                value=f"{fusion_accuracy:.1f}%"
            )

            col2.metric(
                label="CNN + LSTM Fusion Macro-F1",
                value=f"{fusion_f1:.3f}"
            )

            st.caption(
                "Fusion uses synthetically paired image-sensor data. "
                "These results demonstrate the architecture, not "
                "validated real-world agricultural performance."
            )

        # Model comparison table: measured metrics only
        rows = []

        for k, label in [
            ("svm", "SVM (HSI)"),
            ("rgb_cnn", "RGB-CNN"),
            ("cnn", "CNN (HSI)"),
            ("lstm", "LSTM (IoT, simulated)"),
            ("fusion", "CNN + LSTM Fusion"),
        ]:
            if k in rs:
                rows.append({
                    "Model": label,
                    "Measured Accuracy (%)": round(
                        rs[k]["accuracy"] * 100, 1
                    ),
                    "Measured F1": round(
                        rs[k]["macro_f1"], 3
                    ),
                })

        st.subheader("Model Performance — Measured Results")

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        # Model notes
        for k, note in (rs.get("notes") or {}).items():
            st.caption(f"**{k}:** {note}")

        # Evaluation plots
        for fn in (
            "cnn_confusion_matrix.png",
            "cnn_roc_curves.png",
            "cnn_training_history.png",
            "lstm_confusion_matrix.png",
            "fusion_confusion_matrix.png",
        ):
            if (MODEL_DIR / fn).exists():
                st.image(str(MODEL_DIR / fn), caption=fn)

        # Fusion prediction for the currently selected image and sensors
        if (
            pred.fusion is not None
            and hist_w is not None
            and R["mode"] == "CNN"
        ):
            with st.expander(
                "Fusion prototype on the current image + sensor window"
            ):
                pf_ = pred.fusion_predict(R["norm"], hist_w)

                if pf_ is not None:
                    st.write({
                        n: round(float(v), 3)
                        for n, v in zip(R["names"], pf_)
                    })
                else:
                    st.warning(
                        "Fusion prediction is unavailable for this input."
                    )

                st.caption(
                    "Fusion was trained on synthetically paired data, "
                    "so treat this as an architecture demo only."
                )

    else:
        st.info(
            "No evaluation results found. Run `python train.py` "
            "to generate model metrics."
        )

# ---- Report
with tabs[6]:
    payload = dict(filename=chosen, crop=crop,
                   summary={"Field health score": score, "AI diagnosis": R["pred"],
                            "Confidence %": round(R["conf"] * 100, 1), "Diagnosis mode": R["mode"],
                            "Mean NDVI": round(R["stats"]["mean"], 3), "Stressed area %": round(stressed_pct, 1),
                            "Disease-weather risk": f"{risk_level} ({risk})"},
                   indices={k: round(float(v.mean()), 3) for k, v in R["ind"].items()},
                   zones={k: round(v, 1) for k, v in R["zones"].items()},
                   sensors={k: f"{current[k]} {UNITS[k]} ({status[k]})" for k in SENSOR_COLS},
                   alerts=alerts, recommendations=list(dict.fromkeys(recs)))
    st.json(payload, expanded=False)
    d = st.columns(3)
    d[0].download_button("⬇ HTML report", build_html(payload), "agrispectra_report.html", "text/html")
    d[1].download_button("⬇ JSON", build_json(payload), "agrispectra_report.json", "application/json")
    d[2].download_button("⬇ Batch CSV", build_csv([
        {"file": n, "diagnosis": r["pred"], "confidence": round(r["conf"], 4),
         "mean_ndvi": round(r["stats"]["mean"], 4)} for n, r in results.items()]),
        "agrispectra_batch.csv", "text/csv")
