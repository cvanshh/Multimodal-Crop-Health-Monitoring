"""Vegetation indices computed by *wavelength* (not by hard-coded band number)."""
import numpy as np
from data_pipeline import wavelengths
from config import WL_MIN, WL_MAX

ZONE_NAMES = ["Severely stressed", "Stressed", "Moderate", "Good", "Excellent"]
ZONE_COLORS = ["#800000", "#e53935", "#fdd835", "#90ee90", "#1b8a2e"]
ZONE_EDGES = [0.1, 0.3, 0.5, 0.7]       # NDVI thresholds between zones


def band_idx(wl: float, n_bands: int) -> int:
    return int(round((wl - WL_MIN) / (WL_MAX - WL_MIN) * (n_bands - 1)))


def band_mean(cube: np.ndarray, wl: float, half: int = 1) -> np.ndarray:
    n = cube.shape[2]
    i = min(max(band_idx(wl, n), 0), n - 1)
    return cube[:, :, max(i - half, 0): i + half + 1].mean(axis=2)


def to_reflectance(cube: np.ndarray) -> np.ndarray:
    """Scale raw DN to ~0..1 so additive-constant indices (SAVI, PSRI) behave."""
    p99 = np.percentile(cube, 99)
    return cube / p99 if p99 > 1.5 else cube


def compute_indices(cube: np.ndarray) -> dict:
    r = to_reflectance(cube.astype(np.float32))
    b = lambda wl: band_mean(r, wl)  # noqa: E731
    eps = 1e-6
    nir, red, green = b(800), b(670), b(550)
    edge, re740 = b(720), b(740)
    out = {
        "NDVI": (nir - red) / (nir + red + eps),
        "GNDVI": (nir - green) / (nir + green + eps),
        "NDRE": (b(790) - edge) / (b(790) + edge + eps),
        "SAVI": 1.5 * (nir - red) / (nir + red + 0.5),
        "PRI": (b(531) - b(570)) / (b(531) + b(570) + eps),
        "PSRI": (red - b(500)) / (re740 + eps),
        "CIre": nir / (edge + eps) - 1.0,
    }
    return {k: np.clip(np.nan_to_num(v), -1.5, 10.0).astype(np.float32) for k, v in out.items()}


def health_zones(ndvi: np.ndarray) -> np.ndarray:
    return np.digitize(ndvi, ZONE_EDGES).astype(np.int32)       # 0..4


def zone_percentages(ndvi: np.ndarray) -> dict:
    z = health_zones(ndvi)
    tot = z.size
    return {ZONE_NAMES[i]: float((z == i).sum() / tot * 100) for i in range(5)}


def zone_rgb(ndvi: np.ndarray) -> np.ndarray:
    z = health_zones(ndvi)
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in ZONE_COLORS], dtype=np.uint8)
    return pal[z]


def ndvi_stats(ndvi: np.ndarray) -> dict:
    return {
        "min": float(ndvi.min()), "max": float(ndvi.max()),
        "mean": float(ndvi.mean()), "median": float(np.median(ndvi)),
        "std": float(ndvi.std()),
        "p25": float(np.percentile(ndvi, 25)), "p75": float(np.percentile(ndvi, 75)),
    }


def mean_spectrum(cube: np.ndarray):
    """Mean and std reflectance per band -> (wavelengths, mean, std)."""
    r = to_reflectance(cube.astype(np.float32))
    return wavelengths(r.shape[2]), r.mean(axis=(0, 1)), r.std(axis=(0, 1))
