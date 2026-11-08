# pipeline/verify.py — checks a built (or published) site before anyone
# relies on it: the manifest is the app's, every listed tile is there and
# whole, nothing stray is lying around, a full-world build is not
# suspiciously small, and known bright/dark places read as they should.
#
# A site is read through `read(relpath) -> bytes | None` (None = absent), so
# the same checks run on a local directory and on the live GitHub Pages URL.
import json
import math
import os
import random
import re
import http.client
import time
import urllib.error
import urllib.request
import zlib

import numpy as np

from .encode import R0, SCALE, decode
from .glow import GLOW_R0, GLOW_SCALE, decode_glow
from .grid import CELLS, COARSE_DEG, FINE_DEG, tile_key

TILE_KEY_RE = re.compile(r"^[NS]\d{2}[EW]\d{3}$")   # the app's TILE_KEY_RE (src/skyglowload.js)

# A full-world build (80N..70S) of VNL V2 2025 made 4,927 fine + 386 coarse
# tiles. Lit area barely moves year to year, so a build far below that read a
# truncated or wrong raster (a monthly composite, a regional cut, a bad
# gunzip) and must not replace the published year.
MIN_FULL_WORLD = {"fine": 3900, "coarse": 300}


def local_reader(site_dir):
    def read(rel):
        path = os.path.join(site_dir, rel)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read()
    read.site_dir = site_dir
    return read


def http_reader(base, attempts=4, timeout=30, backoff=2.0, opener=None, sleep=time.sleep):
    """Fetches base/rel. A 404 is "absent" (None) at once; a dropped
    connection, timeout, 429 or 5xx is retried with backoff and raises after
    the last attempt — a flaky network must never read as a missing tile.
    A body cut short (IncompleteRead) is retried too: GitHub Pages does it."""
    opener = opener or urllib.request.urlopen
    base = base.rstrip("/")

    def read(rel):
        url = f"{base}/{rel}"
        for i in range(attempts):
            try:
                with opener(url, timeout=timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code != 429 and e.code < 500:
                    raise
                err = e
            except (OSError, http.client.HTTPException) as e:   # dropped, timed out, short body
                err = e
            if i < attempts - 1:
                sleep(backoff * (2 ** i))
        raise RuntimeError(f"{url}: {err} after {attempts} attempts")
    return read


def _tile_bytes(raw):
    b = zlib.decompress(raw)
    if len(b) != CELLS * CELLS:
        raise ValueError(f"{len(b)} bytes, expected {CELLS * CELLS}")
    return np.frombuffer(b, np.uint8).reshape(CELLS, CELLS)


def _fine_radiance(read, lat, lon):
    """Decoded radiance (nW·cm⁻²·sr⁻¹) of the fine cell containing (lat, lon).
    A tile the manifest does not list (or that is simply absent) is dark."""
    if isinstance(read, str):
        read = local_reader(read)
    la = math.floor(lat / FINE_DEG) * FINE_DEG
    lo = math.floor(lon / FINE_DEG) * FINE_DEG
    raw = read(f"fine/{tile_key(la, lo)}.bin")
    if raw is None:
        return 0.0
    b = _tile_bytes(raw)
    per_deg = CELLS / FINE_DEG
    row = min(max(int((la + FINE_DEG - lat) * per_deg), 0), CELLS - 1)   # row 0 = north edge
    col = min(max(int((lon - lo) * per_deg), 0), CELLS - 1)
    return float(decode(int(b[row, col])))


def verify(site, checks):
    """checks: [(name, lat, lon, op, value), ...] with op '>' or '<', each
    compared against the decoded radiance of the fine cell at (lat, lon).
    `site` is a directory or a reader. Returns a list of failure strings."""
    read = local_reader(site) if isinstance(site, str) else site
    failures = []
    for name, lat, lon, op, value in checks:
        r = _fine_radiance(read, lat, lon)
        ok = r > value if op == '>' else r < value
        if not ok:
            failures.append(f"{name} ({lat}, {lon}): {r:.3f} nW·cm⁻²·sr⁻¹, expected {op} {value}")
    return failures


def check_site(read, full_world=True, lat_top=80, lat_bottom=-70, sample=None, seed=0, require_glow=False):
    """Everything the app needs from a site. Returns failure strings (empty
    = publishable). `sample=None` reads every listed tile; an int reads that
    many at random plus the known places' tiles (for the live URL)."""
    if isinstance(read, str):
        read = local_reader(read)
    raw = read("manifest.json")
    if raw is None:
        return ["manifest.json is missing"]
    try:
        m = json.loads(raw)
    except ValueError as e:
        return [f"manifest.json does not parse: {e}"]
    fails = []
    # The same checks the app makes (src/skyglowload.js); any of them failing
    # switches light pollution off for every user.
    if m.get("encoding") != {"scale": SCALE, "r0": R0}:
        fails.append(f"manifest encoding {m.get('encoding')!r} is not scale {SCALE}, r0 {R0}")
    keys = {}
    for kind, deg in (("fine", FINE_DEG), ("coarse", COARSE_DEG)):
        g = m.get(kind)
        if not isinstance(g, dict) or g.get("deg") != deg or g.get("cells") != CELLS \
                or not isinstance(g.get("keys"), list):
            fails.append(f"manifest {kind} grid is not deg {deg}, cells {CELLS} with a key list")
            continue
        ks = g["keys"]
        bad = [k for k in ks if not isinstance(k, str) or not TILE_KEY_RE.match(k)]
        if bad:
            fails.append(f"manifest {kind} has malformed keys, e.g. {bad[:3]}")
        if len(set(ks)) != len(ks):
            fails.append(f"manifest {kind} lists a key twice")
        keys[kind] = [k for k in ks if k not in bad]
        if full_world and len(ks) < MIN_FULL_WORLD[kind]:
            fails.append(f"only {len(ks)} {kind} tiles (a full world has >= {MIN_FULL_WORLD[kind]}): "
                         "truncated or wrong source raster?")
    gkeys = {}
    g = m.get("glow")
    if g is None:
        if require_glow:
            fails.append("manifest has no glow block")
    elif not isinstance(g, dict):
        fails.append("manifest glow is not an object")
    else:
        if g.get("encoding") != {"scale": GLOW_SCALE, "r0": GLOW_R0}:
            fails.append(f"manifest glow encoding {g.get('encoding')!r} is not scale {GLOW_SCALE}, r0 {GLOW_R0}")
        for kind, deg in (("fine", FINE_DEG), ("coarse", COARSE_DEG)):
            gg = g.get(kind)
            if not isinstance(gg, dict) or gg.get("deg") != deg or gg.get("cells") != CELLS \
                    or not isinstance(gg.get("keys"), list):
                fails.append(f"manifest glow {kind} grid is not deg {deg}, cells {CELLS} with a key list")
                continue
            bad = [k for k in gg["keys"] if not isinstance(k, str) or not TILE_KEY_RE.match(k)]
            if bad:
                fails.append(f"manifest glow {kind} has malformed keys, e.g. {bad[:3]}")
            if len(set(gg["keys"])) != len(gg["keys"]):
                fails.append(f"manifest glow {kind} lists a key twice")
            gkeys[kind] = [k for k in gg["keys"] if k not in bad]
    if fails:
        return fails
    # Every listed tile must be there and whole: one listed-but-missing tile
    # fails the app's whole load for every place near it.
    todo = [(kind, k) for kind in ("fine", "coarse") for k in keys[kind]]
    todo += [("glow/" + kind, k) for kind in ("fine", "coarse") for k in gkeys.get(kind, [])]
    places = [p for p in KNOWN_PLACES if lat_bottom <= p[1] < lat_top]
    if sample is not None:
        rng = random.Random(seed)
        want = {("fine", tile_key(math.floor(p[1] / FINE_DEG) * FINE_DEG,
                                  math.floor(p[2] / FINE_DEG) * FINE_DEG)) for p in places}
        if g is not None:
            want |= {("glow/fine", k) for _, k in want}
            want |= {("glow/fine", tile_key(math.floor(p[1] / FINE_DEG) * FINE_DEG,
                                            math.floor(p[2] / FINE_DEG) * FINE_DEG)) for p in GLOW_PINS}
        todo = [t for t in todo if t in want] + rng.sample(todo, min(sample, len(todo)))
    for kind, k in todo:
        rel = f"{kind}/{k}.bin"
        try:
            b = read(rel)
            if b is None:
                fails.append(f"{rel} is listed but missing")
            elif not _tile_bytes(b).any():
                fails.append(f"{rel} is listed but all dark")
        except (zlib.error, ValueError) as e:
            fails.append(f"{rel} is not a {CELLS}x{CELLS} zlib tile: {e}")
        if len(fails) > 20:
            fails.append("... stopping after 20 tile failures")
            return fails
    # Stray tiles (a leftover year, an interrupted copy) — only seeable locally.
    site_dir = getattr(read, "site_dir", None)
    if site_dir:
        listed = [(kind, keys[kind]) for kind in ("fine", "coarse")]
        if g is not None:
            listed += [("glow/" + kind, gkeys.get(kind, [])) for kind in ("fine", "coarse")]
        for kind, ks in listed:
            d = os.path.join(site_dir, *kind.split("/"))
            have = {f[:-4] for f in os.listdir(d) if f.endswith(".bin")} if os.path.isdir(d) else set()
            extra = sorted(have - set(ks))
            if extra:
                fails.append(f"{len(extra)} {kind} tiles not in the manifest, e.g. {extra[:3]}")
    if not fails:
        fails += verify(read, places)
        if g is not None and full_world and lat_top >= 80 and lat_bottom <= -70:
            fails += check_glow_pins(read)
    return fails


def _glow_mag(read, lat, lon):
    """21.9 - 2.5·log10(1 + r) from the decoded fine glow cell containing the
    point; an absent glow tile is r = 0 (darker than class 1)."""
    la = math.floor(lat / FINE_DEG) * FINE_DEG
    lo = math.floor(lon / FINE_DEG) * FINE_DEG
    raw = read(f"glow/fine/{tile_key(la, lo)}.bin")
    r = 0.0
    if raw is not None:
        b = _tile_bytes(raw)
        per_deg = CELLS / FINE_DEG
        row = min(max(int((la + FINE_DEG - lat) * per_deg), 0), CELLS - 1)
        col = min(max(int((lon - lo) * per_deg), 0), CELLS - 1)
        r = float(decode_glow(int(b[row, col])))
    return 21.9 - 2.5 * math.log10(1 + r)


# Measured with the app's glowAt on the 2025 tiles (not derived): the map must
# agree with the 3D sky. Tolerance in magnitudes.
GLOW_PINS = [
    ("Bend", 44.06, -121.31, 19.05),
    ("Fort Rock", 43.36, -121.06, 21.80),
    ("Crater Lake", 42.94, -122.10, 21.90),
]
GLOW_PIN_TOL = 0.1


def check_glow_pins(read, pins=GLOW_PINS, tol=GLOW_PIN_TOL):
    if isinstance(read, str):
        read = local_reader(read)
    fails = []
    for name, lat, lon, want in pins:
        got = _glow_mag(read, lat, lon)
        if abs(got - want) > tol:
            fails.append(f"glow pin {name} ({lat}, {lon}): mag {got:.2f}, expected {want} ± {tol}")
    return fails


KNOWN_PLACES = [
    ("Las Vegas Strip", 36.1147, -115.1728, '>', 50),
    ("Great Basin NP, Wheeler Peak area", 38.9833, -114.3000, '<', 1),
    ("São Paulo centre", -23.5505, -46.6333, '>', 20),
    ("Suva, Fiji", -18.1416, 178.4419, '>', 2),   # the E178 tile next to the date line
    ("Reykjavík", 64.1466, -21.9426, '>', 10),
]
