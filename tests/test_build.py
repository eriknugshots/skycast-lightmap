import json
import zlib

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pipeline.build import ATTRIBUTION, build
from pipeline.grid import Source


def make_tif(path, west, north, res, arr, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(west, north, res, res),
                       nodata=nodata) as d:
        d.write(arr.astype(np.float32), 1)


def tile(path):
    return np.frombuffer(zlib.decompress(path.read_bytes()), np.uint8).reshape(240, 240)


def test_one_lit_cell_makes_one_fine_and_one_coarse_tile(tmp_path):
    arr = np.zeros((240, 240), np.float32)      # lat 46..44, lon -122..-120 at 30″
    arr[10, 20] = 10.0
    tif = tmp_path / "patch.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    out = tmp_path / "site"
    m, nbytes = build(Source(str(tif)), str(out), 2024, "patch.tif", 50, 40, log=lambda *_: None)
    assert m["fine"]["keys"] == ["N44W122"]
    assert m["coarse"]["keys"] == ["N40W130"]
    fine = tile(out / "fine" / "N44W122.bin")
    assert fine[10, 20] == 113 and int(fine.sum()) == 113
    # 10 nW in one of the 25 fine cells of a coarse cell: mean 0.4 → byte 29
    coarse = tile(out / "coarse" / "N40W130.bin")
    assert coarse[98, 196] == 29 and int(coarse.sum()) == 29
    assert nbytes == (out / "fine" / "N44W122.bin").stat().st_size + (out / "coarse" / "N40W130.bin").stat().st_size


def test_native_15_arcsec_cells_are_averaged_in_twos(tmp_path):
    arr = np.zeros((480, 480), np.float32)      # 15″, like EOG's grid
    arr[20:22, 40:42] = [[10.0, 30.0], [0.0, 0.0]]   # fine cell (10, 20): mean 10
    tif = tmp_path / "native.tif"
    make_tif(tif, -122.0, 46.0, 1 / 240, arr)
    m, _ = build(Source(str(tif)), str(tmp_path / "site"), 2024, "native.tif", 50, 40, log=lambda *_: None)
    assert tile(tmp_path / "site" / "fine" / "N44W122.bin")[10, 20] == 113


def test_nodata_and_negative_are_dark(tmp_path):
    arr = np.full((240, 240), -1.5, np.float32)
    arr[0, 0] = -9999.0
    tif = tmp_path / "dark.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr, nodata=-9999.0)
    m, nbytes = build(Source(str(tif)), str(tmp_path / "site"), 2024, "dark.tif", 50, 40, log=lambda *_: None)
    assert m["fine"]["keys"] == [] and m["coarse"]["keys"] == [] and nbytes == 0


def test_manifest(tmp_path):
    arr = np.zeros((240, 240), np.float32)
    arr[0, 0] = 5.0
    tif = tmp_path / "p.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    build(Source(str(tif)), str(tmp_path / "site"), 2024, "p.tif", 50, 40, log=lambda *_: None)
    m = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert m["year"] == 2024 and m["source"] == "p.tif"
    assert m["encoding"] == {"scale": 70.0, "r0": 0.25}
    assert m["fine"]["deg"] == 2 and m["fine"]["cells"] == 240
    assert m["coarse"]["deg"] == 10 and m["coarse"]["cells"] == 240
    assert m["attribution"] == ATTRIBUTION and "CC BY 4.0" in ATTRIBUTION
