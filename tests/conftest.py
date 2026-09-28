import zlib

import numpy as np
import rasterio
from affine import Affine


def make_tif(path, west, north, res, arr, nodata=None, dtype="float32", count=1):
    """A tiny north-up GeoTIFF for tests. `arr` is (H, W) for count=1, or
    (count, H, W) for count>1. Built with Affine's `@` (not rasterio's
    from_origin, whose own `*` triggers a PendingDeprecationWarning)."""
    transform = Affine.translation(west, north) @ Affine.scale(res, -res)
    if count == 1:
        height, width = arr.shape
    else:
        _, height, width = arr.shape
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=count,
                       dtype=dtype, crs="EPSG:4326", transform=transform, nodata=nodata) as d:
        if count == 1:
            d.write(arr.astype(dtype), 1)
        else:
            d.write(arr.astype(dtype))


def tile(path):
    return np.frombuffer(zlib.decompress(path.read_bytes()), np.uint8).reshape(240, 240)
