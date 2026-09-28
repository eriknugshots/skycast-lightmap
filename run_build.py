#!/usr/bin/env python3
"""Build SkyCast's light-pollution tiles from one EOG VNL V2 GeoTIFF.

    .venv/bin/python run_build.py --src data/<file>.tif --year 2024 --out site
"""
import argparse
import os
import sys

from pipeline.build import build
from pipeline.grid import Source


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="the median-masked GeoTIFF (gunzipped)")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out", default="site")
    p.add_argument("--lat-top", type=int, default=80)
    p.add_argument("--lat-bottom", type=int, default=-70)
    a = p.parse_args(argv)
    src = Source(a.src)
    print(f"{os.path.basename(a.src)}: {src.ds.width}x{src.ds.height} px, {src.per_deg}/deg, "
          f"west {src.west!r}, north {src.north!r}, lon -180 at column {src.col_west:.4f}, nodata {src.nodata!r}")
    m, nbytes = build(src, a.out, a.year, os.path.basename(a.src), a.lat_top, a.lat_bottom)
    print(f"done: {len(m['fine']['keys'])} fine + {len(m['coarse']['keys'])} coarse tiles, {nbytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
