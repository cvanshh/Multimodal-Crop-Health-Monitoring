"""Display helpers + synthetic demo cubes (so the dashboard works without any dataset)."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from config import IMG_SIZE, NUM_BANDS
from data_pipeline import wavelengths
from indices import band_mean


def _stretch(img):
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def true_color(cube):
    """Approx RGB from ~650 / 550 / 470 nm."""
    rgb = np.dstack([band_mean(cube, 650), band_mean(cube, 550), band_mean(cube, 470)])
    return (_stretch(rgb) * 255).astype(np.uint8)


def color_infrared(cube):
    """CIR false colour: NIR(800) -> R, Red(670) -> G, Green(550) -> B. Healthy plants glow red."""
    rgb = np.dstack([band_mean(cube, 800), band_mean(cube, 670), band_mean(cube, 550)])
    return (_stretch(rgb) * 255).astype(np.uint8)


def overlay_heatmap(base_rgb, heat, alpha=0.45):
    h = cv2.resize(heat.astype(np.float32), (base_rgb.shape[1], base_rgb.shape[0]),
                   interpolation=cv2.INTER_CUBIC)
    h = np.uint8(255 * np.clip(h, 0, 1))
    h = cv2.cvtColor(cv2.applyColorMap(h, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, h, alpha, 0)


def upscale(img, factor=6, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    out = cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)
    return np.clip(out, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out


# ---------------- synthetic data ----------------
def _sig(x):
    return 1 / (1 + np.exp(-x))


def _profile(wl, nir, red, edge_pos, edge_w):
    green_bump = 0.06 * np.exp(-((wl - 550) / 28) ** 2)
    return 0.035 + green_bump + (red - 0.035) * np.exp(-((wl - 670) / 25) ** 2) \
        + nir * _sig((wl - edge_pos) / edge_w)


def synthetic_cube(kind="healthy", size=IMG_SIZE, bands=NUM_BANDS, seed=0):
    """kind in {'healthy','rust','mixed'}. Returns float32 cube (size,size,bands) of reflectance."""
    rng = np.random.default_rng(seed)
    wl = wavelengths(bands)
    healthy = _profile(wl, nir=0.50, red=0.035, edge_pos=715, edge_w=14)
    rusty = _profile(wl, nir=0.33, red=0.10, edge_pos=728, edge_w=24)
    field = gaussian_filter(rng.normal(size=(size, size)), sigma=size / 9)
    field = (field - field.min()) / (field.max() - field.min())
    if kind == "healthy":
        sev = 0.04 * field
    elif kind == "rust":
        sev = np.clip(0.55 + 0.6 * (field - 0.5), 0, 1)
    else:
        sev = np.clip((field - 0.55) * 3.0, 0, 1)
    sev = sev[..., None]
    cube = healthy * (1 - sev) + rusty * sev
    cube *= (1 + 0.05 * gaussian_filter(rng.normal(size=(size, size)), 2))[..., None]
    cube += rng.normal(0, 0.006, cube.shape)
    return np.clip(cube, 0, 1).astype(np.float32)
