# pipeline/glow.py — the zenith sky glow the app's glowAt (src/skyglow.js)
# computes at a point, computed here for every cell of a band by FFT, and the
# one-byte encoding the map's "Dark sky" layer decodes. Change the constants
# with skyglow.js (SKYGLOW_C, kernel, KM_LAT/KM_LON, 300 km cut) and the
# manifest's "glow.encoding" block together.
import math

import numpy as np

GLOW_SCALE = 40.0
GLOW_R0 = 0.002
SKYGLOW_C = 0.0576
FAR_KM = 300.0
KM_LAT, KM_LON = 110.57, 111.32
# Bortle class 1 starts at m' = 21.99 (m' = 22.0 - 2.5·log10(1 + r)).
CLASS1_R = 10.0 ** (0.4 * (22.0 - 21.99)) - 1.0


def encode_glow(r):
    """r → uint8: round(SCALE·log10(1 + r/R0)), half up; negative/NaN/inf → 0."""
    a = np.nan_to_num(np.asarray(r, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, None)
    return np.clip(np.floor(GLOW_SCALE * np.log10(1.0 + a / GLOW_R0) + 0.5), 0, 255).astype(np.uint8)


def decode_glow(b):
    return GLOW_R0 * (10.0 ** (np.asarray(b, dtype=np.float64) / GLOW_SCALE) - 1.0)


def _kernel(cell_deg, lat_c, width):
    """Zenith kernel K(d) on the (row, col) offsets within 300 km, d from the
    flat map at the band-centre latitude; 0 beyond 300 km."""
    hr = math.ceil(FAR_KM / (cell_deg * KM_LAT))
    hc = min(math.ceil(FAR_KM / (cell_deg * KM_LON * math.cos(math.radians(lat_c)))), (width - 1) // 2)
    dy = np.arange(-hr, hr + 1)[:, None] * cell_deg * KM_LAT
    dx = np.arange(-hc, hc + 1)[None, :] * cell_deg * KM_LON * math.cos(math.radians(lat_c))
    d = np.maximum(np.hypot(dy, dx), 1.0)
    k = d ** -2.5 * np.exp(-d / 100.0)
    k[np.hypot(dy, dx) > FAR_KM] = 0.0
    return k, hr, hc


def glow_band(fine_radiance_rows, top_lat, cell_deg, margin_rows):
    """Added-light ratio r = SKYGLOW_C·Σ R·area·K(d) for one latitude band.

    fine_radiance_rows: (band_rows + 2·margin_rows, 360/cell_deg) radiance for
    the full longitude circle, margin rows included (north margin first);
    top_lat is the BAND's top edge. Returns r for the band's own rows. Cell
    area and the longitude scale both use the band-centre cosine."""
    a = np.asarray(fine_radiance_rows, dtype=np.float64)
    n, width = a.shape
    band_rows = n - 2 * margin_rows
    lat_c = top_lat - band_rows * cell_deg / 2.0
    area = cell_deg * cell_deg * KM_LAT * KM_LON * math.cos(math.radians(lat_c))
    k, hr, hc = _kernel(cell_deg, lat_c, width)
    length = n + 2 * hr  # zero-padded rows: linear in latitude, circular in longitude
    kk = np.zeros((length, width))
    rows = np.arange(-hr, hr + 1) % length
    cols = np.arange(-hc, hc + 1) % width
    kk[np.ix_(rows, cols)] = k
    e = np.zeros((length, width))
    e[:n] = a * area
    conv = np.fft.irfft2(np.fft.rfft2(e) * np.fft.rfft2(kk), s=(length, width))
    out = SKYGLOW_C * conv[margin_rows:margin_rows + band_rows]
    out[out < 1e-12 * max(float(out.max()), 1e-300)] = 0.0  # FFT round-off, not light
    return out
