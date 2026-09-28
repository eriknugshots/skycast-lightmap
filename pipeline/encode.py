# pipeline/encode.py — the one-byte radiance encoding SkyCast decodes
# (src/skyglow.js encodeRadiance/decodeByte). Change both together, and the
# manifest's "encoding" block with them, or the app refuses the tiles.
import zlib

import numpy as np

SCALE = 70.0
R0 = 0.25


def encode(radiance):
    """Radiance (nW·cm⁻²·sr⁻¹) → uint8: round(SCALE·log10(1 + R/R0)), 0 = unlit.
    Rounds half UP like JavaScript's Math.round (numpy's rint rounds to even)."""
    r = np.nan_to_num(np.asarray(radiance, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, 0.0, None)
    return np.clip(np.floor(SCALE * np.log10(1.0 + r / R0) + 0.5), 0, 255).astype(np.uint8)


def decode(b):
    return R0 * (10.0 ** (np.asarray(b, dtype=np.float64) / SCALE) - 1.0)


def pack(tile_bytes):
    """zlib (RFC 1950): what the app's DecompressionStream('deflate') reads."""
    return zlib.compress(np.ascontiguousarray(tile_bytes, dtype=np.uint8).tobytes(), 9)
