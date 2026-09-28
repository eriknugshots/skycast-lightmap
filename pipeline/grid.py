# pipeline/grid.py — tile names and the source raster, read in latitude bands.
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


class Source:
    """A plate carrée radiance raster read in whole-world latitude bands,
    zero wherever it has no data (VNL spans 75°N–65°S; the tests use a patch)."""

    def __init__(self, path):
        import rasterio
        self.ds = rasterio.open(path)
        t = self.ds.transform
        self.res = t.a
        self.per_deg = int(round(1.0 / self.res))
        if abs(self.per_deg * self.res - 1.0) > 1e-9 or abs(t.e + t.a) > 1e-12 or t.b or t.d:
            raise ValueError(f"not a north-up grid with whole cells per degree: {t}")
        self.west, self.north = t.c, t.f
        # Where lon -180 and each whole-degree latitude fall, in source pixels.
        # EOG's grids may sit half a pixel off whole degrees; the build rounds
        # and logs the residual so a misread grid cannot pass unnoticed.
        self.col_west = (-180.0 - self.west) / self.res
        self.nodata = self.ds.nodata

    def read_band(self, lat_top, deg):
        """(deg·per_deg) × (360·per_deg) float32 from lon -180 eastward, or None
        if the band misses the raster entirely."""
        from rasterio.windows import Window
        n, width = deg * self.per_deg, 360 * self.per_deg
        r0 = int(round((self.north - lat_top) / self.res))
        c0 = int(round(self.col_west))
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
