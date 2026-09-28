# pipeline/build.py — the whole world in 2° latitude bands: each band's fine
# tiles as it is read, and a 10° band's coarse tiles once its five fine bands
# are in. Only tiles with a lit cell are written; the manifest lists them.
import json
import os

import numpy as np

from .encode import R0, SCALE, encode, pack
from .grid import CELLS, COARSE_DEG, FINE_DEG, FINE_PER_COARSE, block_mean, tile_key

ATTRIBUTION = ("Night lights: Earth Observation Group, Payne Institute for Public Policy "
               "(VIIRS Nighttime Lights, Annual VNL V2, median-masked), CC BY 4.0")


def _write(out_dir, kind, key, b):
    path = os.path.join(out_dir, kind, f"{key}.bin")
    with open(path, "wb") as f:
        f.write(pack(b))
    return os.path.getsize(path)


def build(src, out_dir, year, source_name, lat_top=80, lat_bottom=-70, log=print):
    if lat_top % COARSE_DEG or lat_bottom % COARSE_DEG or lat_top <= lat_bottom:
        raise ValueError("lat_top and lat_bottom must be multiples of 10, top above bottom")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise ValueError(f"{out_dir} already has files in it — delete it or pick a different --out; "
                          "a rerun must never publish last year's leftover tiles")
    native_per_fine = src.per_deg * FINE_DEG // CELLS          # 2 for EOG's 15″ grid
    if native_per_fine < 1 or native_per_fine * CELLS != src.per_deg * FINE_DEG:
        raise ValueError(f"{src.per_deg} px/deg does not divide into {CELLS} cells per {FINE_DEG}°")
    os.makedirs(os.path.join(out_dir, "fine"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "coarse"), exist_ok=True)
    fine_w = 360 * CELLS // FINE_DEG                           # 43200 fine cells round the world
    coarse_rows_per_fine_band = CELLS // FINE_PER_COARSE       # 48
    fine_keys, coarse_keys, nbytes = [], [], 0
    for ctop in range(lat_top, lat_bottom, -COARSE_DEG):
        coarse_rows = []
        for ftop in range(ctop, ctop - COARSE_DEG, -FINE_DEG):
            band = src.read_band(ftop, FINE_DEG)
            if band is None:
                coarse_rows.append(np.zeros((coarse_rows_per_fine_band, fine_w // FINE_PER_COARSE)))
                continue
            band = np.clip(np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
            fine = block_mean(band, native_per_fine)           # CELLS × 43200, radiance at 30″
            for i in range(360 // FINE_DEG):
                b = encode(fine[:, i * CELLS:(i + 1) * CELLS])
                if b.any():
                    key = tile_key(ftop - FINE_DEG, -180 + i * FINE_DEG)
                    nbytes += _write(out_dir, "fine", key, b)
                    fine_keys.append(key)
            coarse_rows.append(block_mean(fine, FINE_PER_COARSE))   # 48 × 8640 at 2.5′: mean of fine cells
        coarse = np.vstack(coarse_rows)                        # CELLS × 8640
        for j in range(360 // COARSE_DEG):
            b = encode(coarse[:, j * CELLS:(j + 1) * CELLS])
            if b.any():
                key = tile_key(ctop - COARSE_DEG, -180 + j * COARSE_DEG)
                nbytes += _write(out_dir, "coarse", key, b)
                coarse_keys.append(key)
        log(f"{ctop:>4}..{ctop - COARSE_DEG:<4} {len(fine_keys):>6} fine {len(coarse_keys):>5} coarse {nbytes / 1e6:8.1f} MB")
    manifest = {
        "year": int(year),
        "source": source_name,
        "encoding": {"scale": SCALE, "r0": R0},
        "fine": {"deg": FINE_DEG, "cells": CELLS, "keys": sorted(fine_keys)},
        "coarse": {"deg": COARSE_DEG, "cells": CELLS, "keys": sorted(coarse_keys)},
        "attribution": ATTRIBUTION,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    return manifest, nbytes
