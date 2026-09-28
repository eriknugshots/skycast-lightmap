#!/usr/bin/env python3
"""Build SkyCast's light-pollution tiles from one EOG VNL V2 GeoTIFF.

    .venv/bin/python run_build.py --src data/<file>.tif --year 2024 --out site
"""
import argparse
import os
import sys

from pipeline.build import build
from pipeline.grid import Source
from pipeline.verify import KNOWN_PLACES, verify


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
          f"west {src.west!r}, north {src.north!r}")
    print(f"  lon -180: frac {src.col180_frac:.9f} px -> col180 {src.col180}; "
          f"lat 90: frac {src.row90_frac:.9f} px -> row90 {src.row90}")
    print(f"  compress {src.ds.profile.get('compress')!r}, block {src.ds.block_shapes[0]}, "
          f"tiled {src.ds.profile.get('tiled')!r}, nodata {src.nodata!r}")
    m, nbytes = build(src, a.out, a.year, os.path.basename(a.src), a.lat_top, a.lat_bottom)
    print(f"done: {len(m['fine']['keys'])} fine + {len(m['coarse']['keys'])} coarse tiles, {nbytes / 1e6:.1f} MB")
    if a.lat_top == 80 and a.lat_bottom == -70:
        failed = False
        for check in KNOWN_PLACES:
            name = check[0]
            failures = verify(a.out, [check])
            if failures:
                print(f"  verify FAIL: {failures[0]}")
                failed = True
            else:
                print(f"  verify ok:   {name}")
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
