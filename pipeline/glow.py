# pipeline/glow.py — the zenith sky glow the app's glowAt (src/skyglow.js)
# computes at a point, computed here for every cell of a band by FFT, and the
# one-byte encoding the map's "Dark sky" layer decodes. Change the constants
# with skyglow.js (SKYGLOW_C, kernel, KM_LAT/KM_LON, 300 km cut) and the
# manifest's "glow.encoding" block together.
import json
import math
import os
import zlib

import numpy as np

from .encode import decode, pack
from .grid import CELLS, COARSE_DEG, FINE_DEG, FINE_PER_COARSE, block_mean, tile_key

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


def _fast_len(n):
    """Smallest 5-smooth integer >= n (numpy has no next_fast_len)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def glow_band(fine_radiance_rows, top_lat, cell_deg, margin_rows):
    """Added-light ratio r = SKYGLOW_C·Σ R·area·K(d) for one latitude band.

    fine_radiance_rows: (band_rows + 2·margin_rows, 360/cell_deg) radiance for
    the full longitude circle, margin rows included (north margin first);
    top_lat is the BAND's top edge. Returns r for the band's own rows. Each SOURCE row is weighted
    by its own cos(lat) (glowAt's per-cell area); only the kernel's longitude
    km scale uses the band-centre cosine (glowAt's single viewer cosLat)."""
    a = np.asarray(fine_radiance_rows, dtype=np.float64)
    n, width = a.shape
    band_rows = n - 2 * margin_rows
    lat_c = top_lat - band_rows * cell_deg / 2.0
    k, hr, hc = _kernel(cell_deg, lat_c, width)
    if margin_rows < hr:
        raise ValueError(f"margin_rows {margin_rows} < kernel reach {hr} rows: far sources would be dropped")
    row_lat = top_lat + margin_rows * cell_deg - (np.arange(n) + 0.5) * cell_deg
    area = cell_deg * cell_deg * KM_LAT * KM_LON * np.cos(np.radians(row_lat))
    length = _fast_len(n + 2 * hr)  # >= n + 2hr: zero-padded rows, linear in latitude, circular in longitude
    kk = np.zeros((length, width))
    kk[np.ix_(np.arange(-hr, hr + 1) % length, np.arange(-hc, hc + 1) % width)] = k
    fk = np.fft.rfft2(kk)
    del kk
    e = np.zeros((length, width))
    e[:n] = a * area[:, None]
    fe = np.fft.rfft2(e)
    del e
    fe *= fk
    del fk
    conv = np.fft.irfft2(fe, s=(length, width))
    del fe
    out = SKYGLOW_C * conv[margin_rows:margin_rows + band_rows]
    out[out < 1e-12 * max(float(out.max()), 1e-300)] = 0.0  # FFT round-off, not light
    return out


def _read_fine_band(out_dir, keys, ftop, width):
    """Decoded radiance (CELLS x width) of the published fine tiles in the 2°
    band whose top edge is ftop; an unlisted tile is dark."""
    band = np.zeros((CELLS, width))
    lat = ftop - FINE_DEG
    for i in range(360 // FINE_DEG):
        key = tile_key(lat, -180 + i * FINE_DEG)
        if key not in keys:
            continue
        with open(os.path.join(out_dir, "fine", f"{key}.bin"), "rb") as f:
            b = np.frombuffer(zlib.decompress(f.read()), np.uint8).reshape(CELLS, CELLS)
        band[:, i * CELLS:(i + 1) * CELLS] = decode(b)
    return band


def _write_glow(out_dir, kind, key, b):
    with open(os.path.join(out_dir, "glow", kind, f"{key}.bin"), "wb") as f:
        f.write(pack(b))


def build_glow(out_dir, lat_top=80, lat_bottom=-70, log=print):
    """Glow tiles from the build's OWN published fine tiles (the quantised
    bytes the app's 3D sky reads), one 2° band at a time. Writes
    out_dir/glow/fine and glow/coarse (a tile only if some cell is brighter
    than Bortle class 1) and returns the manifest's "glow" block."""
    if lat_top % COARSE_DEG or lat_bottom % COARSE_DEG or lat_top <= lat_bottom:
        raise ValueError("lat_top and lat_bottom must be multiples of 10, top above bottom")
    if os.path.exists(os.path.join(out_dir, "glow")):
        raise ValueError(f"{out_dir}/glow already exists — remove it first; a rerun must never keep stale tiles")
    with open(os.path.join(out_dir, "manifest.json")) as f:
        keys = set(json.load(f)["fine"]["keys"])
    os.makedirs(os.path.join(out_dir, "glow", "fine"))
    os.makedirs(os.path.join(out_dir, "glow", "coarse"))
    width = 360 * CELLS // FINE_DEG
    cell_deg = FINE_DEG / CELLS
    margin = math.ceil(FAR_KM / (cell_deg * KM_LAT))
    reach = math.ceil(margin / CELLS)                       # fine bands the margin spans
    cache = {}

    def band(ftop):
        if ftop not in cache:
            cache[ftop] = _read_fine_band(out_dir, keys, ftop, width)
        return cache[ftop]

    fine_keys, coarse_keys = [], []
    for ctop in range(lat_top, lat_bottom, -COARSE_DEG):
        rs = []
        for ftop in range(ctop, ctop - COARSE_DEG, -FINE_DEG):
            rows = np.vstack([band(ftop + FINE_DEG * j) for j in range(reach, -reach - 1, -1)])
            a0 = reach * CELLS - margin
            r = glow_band(rows[a0:a0 + CELLS + 2 * margin], ftop, cell_deg, margin)
            del rows
            for old in [t for t in cache if t > ftop + FINE_DEG * reach - FINE_DEG]:
                del cache[old]
            for i in range(360 // FINE_DEG):
                cell = r[:, i * CELLS:(i + 1) * CELLS]
                if (cell > CLASS1_R).any():
                    key = tile_key(ftop - FINE_DEG, -180 + i * FINE_DEG)
                    _write_glow(out_dir, "fine", key, encode_glow(cell))
                    fine_keys.append(key)
            rs.append(block_mean(r, FINE_PER_COARSE))       # mean r of the 5x5 fine cells
        coarse = np.vstack(rs)
        for j in range(360 // COARSE_DEG):
            cell = coarse[:, j * CELLS:(j + 1) * CELLS]
            if (cell > CLASS1_R).any():
                key = tile_key(ctop - COARSE_DEG, -180 + j * COARSE_DEG)
                _write_glow(out_dir, "coarse", key, encode_glow(cell))
                coarse_keys.append(key)
        log(f"glow {ctop:>4}..{ctop - COARSE_DEG:<4} {len(fine_keys):>6} fine {len(coarse_keys):>5} coarse")
    return {
        "encoding": {"scale": GLOW_SCALE, "r0": GLOW_R0},
        "fine": {"deg": FINE_DEG, "cells": CELLS, "keys": sorted(fine_keys)},
        "coarse": {"deg": COARSE_DEG, "cells": CELLS, "keys": sorted(coarse_keys)},
    }
