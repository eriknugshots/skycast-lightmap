#!/usr/bin/env python3
"""Build SkyCast's light-pollution tiles from one EOG VNL V2 GeoTIFF, or
check a built / published site.

    .venv/bin/python run_build.py --src data/<file>.tif --year 2024 --out build-2024
    .venv/bin/python run_build.py --check build-2024
    .venv/bin/python run_build.py --check-url https://eriknugshots.github.io/skycast-lightmap

A build is written to <out>.building and only renamed to <out> once every
check passes, so <out> existing means it is fit to publish; a failed build
stays in <out>.building for inspection and nothing published is touched.
"""
import argparse
import json
import os
import sys

from pipeline.build import build
from pipeline.glow import build_glow
from pipeline.manifest import write_manifest
from pipeline.grid import Source
from pipeline.verify import check_site, http_reader, local_reader

PAGES_URL = "https://eriknugshots.github.io/skycast-lightmap"


def _report(fails, what):
    for f in fails:
        print(f"  check FAIL: {f}")
    print(f"{what}: {'FAILED' if fails else 'ok'}")
    return 1 if fails else 0


def glow_only(path, lat_top, lat_bottom):
    """Adds glow tiles to an existing build from its own fine tiles. Never
    touches site/ (the gh-pages working copy): run it on a copy."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.realpath(path) == os.path.realpath(os.path.join(here, "site")):
        print("refusing to write into site/ — copy it first (cp -R site build-glow) and run on the copy")
        return 1
    mpath = os.path.join(path, "manifest.json")
    if not os.path.isfile(mpath):
        print(f"{path}/manifest.json is missing — not a build directory")
        return 1
    with open(mpath) as f:
        manifest = json.load(f)
    try:
        manifest["glow"] = build_glow(path, lat_top, lat_bottom)
    except ValueError as e:
        print(e)
        return 1
    write_manifest(path, manifest)
    full = lat_top == 80 and lat_bottom == -70
    fails = check_site(local_reader(path), full_world=full, lat_top=lat_top, lat_bottom=lat_bottom,
                       require_glow=True)
    return _report(fails, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", help="the median-masked GeoTIFF (gunzipped)")
    p.add_argument("--year", type=int)
    p.add_argument("--out", default="site")
    p.add_argument("--lat-top", type=int, default=80)
    p.add_argument("--lat-bottom", type=int, default=-70)
    p.add_argument("--glow-only", metavar="DIR",
                   help="add glow tiles + the manifest glow block to an existing build directory "
                        "(e.g. a COPY of site/; never site/ itself) and check it")
    p.add_argument("--check", metavar="DIR", help="check a built site directory and exit")
    p.add_argument("--check-url", metavar="URL", nargs="?", const=PAGES_URL,
                   help=f"check the live site (default {PAGES_URL}) and exit")
    p.add_argument("--sample", type=int, default=300, help="tiles to read for --check-url")
    a = p.parse_args(argv)

    if a.glow_only:
        return glow_only(a.glow_only, a.lat_top, a.lat_bottom)
    if a.check:
        return _report(check_site(local_reader(a.check), require_glow=True), a.check)
    if a.check_url:
        try:
            fails = check_site(http_reader(a.check_url), sample=a.sample, require_glow=True)
        except RuntimeError as e:   # the network gave up: unknown is not ok
            fails = [str(e)]
        return _report(fails, a.check_url)
    if not a.src or a.year is None:
        p.error("--src and --year are required to build")

    out = a.out.rstrip("/")
    stage = out + ".building"
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        p.error(f"{out} already exists — build to a new --out; a rerun must never mix years")
    if os.path.exists(stage):
        p.error(f"{stage} is left from an earlier run — delete it first")
    src = Source(a.src)
    print(f"{os.path.basename(a.src)}: {src.ds.width}x{src.ds.height} px, {src.per_deg}/deg, "
          f"west {src.west!r}, north {src.north!r}")
    print(f"  lon -180: frac {src.col180_frac:.9f} px -> col180 {src.col180}; "
          f"lat 90: frac {src.row90_frac:.9f} px -> row90 {src.row90}")
    print(f"  compress {src.ds.profile.get('compress')!r}, block {src.ds.block_shapes[0]}, "
          f"tiled {src.ds.profile.get('tiled')!r}, nodata {src.nodata!r}")
    m, nbytes = build(src, stage, a.year, os.path.basename(a.src), a.lat_top, a.lat_bottom, glow=True)
    print(f"done: {len(m['fine']['keys'])} fine + {len(m['coarse']['keys'])} coarse tiles, {nbytes / 1e6:.1f} MB")
    full = a.lat_top == 80 and a.lat_bottom == -70
    if not full:
        print("  partial extent: tile-count floors skipped — NOT a site to publish")
    fails = check_site(local_reader(stage), full_world=full, lat_top=a.lat_top, lat_bottom=a.lat_bottom,
                       require_glow=True)
    if _report(fails, stage):
        print(f"  left in {stage}; {out} not written")
        return 1
    if os.path.isdir(out):
        os.rmdir(out)          # the empty directory checked above
    os.rename(stage, out)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
