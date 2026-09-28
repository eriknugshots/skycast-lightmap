# pipeline/grid.py — tile names and the source raster, read in latitude bands.
import math

import numpy as np

FINE_DEG, COARSE_DEG, CELLS = 2, 10, 240
# A coarse cell (10°/240 = 2.5′) is exactly 5 fine cells (2°/240 = 30″) a side.
FINE_PER_COARSE = (CELLS // FINE_DEG) // (CELLS // COARSE_DEG)


def tile_key(sw_lat, sw_lon):
    """SW corner → "N44W122", the app's tileKey (src/skyglow.js)."""
    la, lo = int(sw_lat), int(sw_lon)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def block_mean(a, f):
    """Mean over f×f blocks (both sides must divide by f)."""
    if f == 1:
        return a.astype(np.float64)
    h, w = a.shape
    return a.reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float64)


def align(x, what):
    """A fractional source-pixel index x, snapped once to a whole- or
    half-pixel grid and returned as the integer pixel index.

    EOG's VNL rasters are pixel-centred, so a whole-degree line (lon -180,
    lat 90) sits half a pixel off the corner-based transform's own grid: x
    lands at k + 0.5, and we take the pixel centred INSIDE that cell —
    east/south of the whole-degree edge — via floor(x) + 1. A whole-pixel
    grid (x within 1e-6 of an integer) is returned as round(x). Anything
    else means the origin was misread, or isn't on a whole/half-pixel grid
    at all: raise rather than silently misplace every tile.
    """
    f = x - math.floor(x)
    if min(f, 1 - f) < 1e-6:
        return round(x)
    if abs(f - 0.5) < 1e-6:
        return math.floor(x) + 1
    raise ValueError(f"{what}: grid is {f:.9f} px off whole degrees")


class Source:
    """A plate carrée radiance raster read in whole-world latitude bands,
    zero wherever it has no data (VNL spans 75°N–65°S; the tests use a patch)."""

    def __init__(self, path):
        import rasterio
        self.ds = rasterio.open(path)
        if self.ds.count != 1:
            raise ValueError(f"expected a single-band raster, got {self.ds.count} bands")
        if self.ds.dtypes[0] != "float32":
            raise ValueError(f"expected float32 radiance, got dtype {self.ds.dtypes[0]!r}")
        if self.ds.crs is None or not self.ds.crs.is_geographic:
            raise ValueError(f"expected a geographic (lat/lon) CRS, got {self.ds.crs!r}")
        t = self.ds.transform
        if not t.a > 0:
            raise ValueError(f"expected north-up with positive x resolution, got {t}")
        self.res = t.a
        self.per_deg = int(round(1.0 / self.res))
        if abs(self.per_deg * self.res - 1.0) > 1e-7 or abs(t.e + t.a) > 1e-12 or t.b or t.d:
            raise ValueError(f"not a north-up grid with whole cells per degree: {t}")
        self.west, self.north = t.c, t.f
        # Where lon -180 and lat 90 fall, in source pixels — decided once,
        # deterministically, here. read_band must never re-derive this from
        # floats per band: a fraction near k + 0.5 can round to k or k + 1
        # inconsistently across bands, duplicating or dropping a source row
        # at the seam between two bands.
        col_x = (-180.0 - self.west) / self.res
        row_x = (self.north - 90.0) / self.res
        self.col180_frac = col_x - math.floor(col_x)
        self.row90_frac = row_x - math.floor(row_x)
        self.col180 = align(col_x, "lon -180")
        self.row90 = align(row_x, "lat 90")
        self.nodata = self.ds.nodata

    def read_band(self, lat_top, deg):
        """(deg·per_deg) × (360·per_deg) float32 from lon -180 eastward, or None
        if the band misses the raster entirely."""
        from rasterio.windows import Window
        n, width = deg * self.per_deg, 360 * self.per_deg
        r0 = self.row90 + (90 - lat_top) * self.per_deg
        c0 = self.col180
        rs, re = max(r0, 0), min(r0 + n, self.ds.height)
        cs, ce = max(c0, 0), min(c0 + width, self.ds.width)
        if rs >= re or cs >= ce:
            return None
        a = self.ds.read(1, window=Window(cs, rs, ce - cs, re - rs)).astype(np.float32)
        if self.nodata is not None:
            a[a == self.nodata] = 0.0
        out = np.zeros((n, width), dtype=np.float32)
        out[rs - r0:re - r0, cs - c0:ce - c0] = a
        return out
