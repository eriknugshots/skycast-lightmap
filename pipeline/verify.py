# pipeline/verify.py — sanity checks against known bright/dark places, run
# by run_build.py after a full-world (default-extent) build.
import math
import os
import zlib

import numpy as np

from .encode import decode
from .grid import CELLS, FINE_DEG, tile_key


def _fine_radiance(site_dir, lat, lon):
    """Decoded radiance (nW·cm⁻²·sr⁻¹) of the fine cell containing (lat, lon).
    A tile the manifest does not list (or that is simply absent) is dark."""
    la = math.floor(lat / FINE_DEG) * FINE_DEG
    lo = math.floor(lon / FINE_DEG) * FINE_DEG
    path = os.path.join(site_dir, "fine", f"{tile_key(la, lo)}.bin")
    if not os.path.isfile(path):
        return 0.0
    with open(path, "rb") as f:
        b = np.frombuffer(zlib.decompress(f.read()), np.uint8).reshape(CELLS, CELLS)
    per_deg = CELLS / FINE_DEG
    row = min(max(int((la + FINE_DEG - lat) * per_deg), 0), CELLS - 1)   # row 0 = north edge
    col = min(max(int((lon - lo) * per_deg), 0), CELLS - 1)
    return float(decode(int(b[row, col])))


def verify(site_dir, checks):
    """checks: [(name, lat, lon, op, value), ...] with op '>' or '<', each
    compared against the decoded radiance of the fine cell at (lat, lon).
    Returns a list of failure strings (empty when every check passes)."""
    failures = []
    for name, lat, lon, op, value in checks:
        r = _fine_radiance(site_dir, lat, lon)
        ok = r > value if op == '>' else r < value
        if not ok:
            failures.append(f"{name} ({lat}, {lon}): {r:.3f} nW·cm⁻²·sr⁻¹, expected {op} {value}")
    return failures


KNOWN_PLACES = [
    ("Las Vegas Strip", 36.1147, -115.1728, '>', 50),
    ("Great Basin NP, Wheeler Peak area", 38.9833, -114.3000, '<', 1),
    ("São Paulo centre", -23.5505, -46.6333, '>', 20),
    ("Suva, Fiji", -18.1416, 178.4419, '>', 2),   # the E178 tile next to the date line
    ("Reykjavík", 64.1466, -21.9426, '>', 10),
]
