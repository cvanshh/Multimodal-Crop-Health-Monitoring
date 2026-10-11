"""Generate SYNTHETIC hyperspectral TIFFs + a sensor CSV for testing the pipeline.

    python make_synthetic_data.py --paper_counts --size 64      # 500 images like the paper
    python train.py --data_dir archive_synth/train --epochs 5

WARNING: synthetic data only proves the code runs. Never mix it into real training/test data and
never report accuracy measured on it as a research result.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from imaging import synthetic_cube
from iot_data import FEATURES, RANGES
from sensors import simulate_history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="archive_synth")
    ap.add_argument("--n_per_class", type=int, default=100, help="same count for every class")
    ap.add_argument("--paper_counts", action="store_true", help="Health 160, Rust 185, Other 155 (= 500, as in the paper)")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--bands", type=int, default=125)
    a = ap.parse_args()

    kinds = {"Health": "healthy", "Rust": "rust", "Other": "mixed"}
    counts = {"Health": 160, "Rust": 185, "Other": 155} if a.paper_counts else {c: a.n_per_class for c in kinds}
    for cls, kind in kinds.items():
        d = Path(a.out) / "train" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(counts[cls]):
            cube = synthetic_cube(kind, size=a.size, bands=a.bands, seed=i + 1000 * list(kinds).index(cls))
            arr = np.transpose((cube * 4000).astype("uint16"), (2, 0, 1))
            with rasterio.open(d / f"synth ({i + 1}).tif", "w", driver="GTiff", height=a.size, width=a.size,
                               count=a.bands, dtype="uint16") as dst:
                dst.write(arr)
        print(f"wrote {counts[cls]} images -> {d}")

    # paper-style sensor table: 1000 uniform records + rule label
    rng = np.random.default_rng(42)
    import pandas as pd
    df = pd.DataFrame({f: rng.uniform(*RANGES[f], 1000).round(2) for f in FEATURES})
    df["label"] = np.where((df.soil_moisture > 30) & (df.humidity > 50), "Healthy", "Unhealthy")
    df.to_csv(Path(a.out) / "iot_sensor_data.csv", index=False)
    simulate_history(60).to_csv(Path(a.out) / "sensor_history_for_dashboard.csv", index=False)
    print("wrote sensor CSVs")


if __name__ == "__main__":
    main()
