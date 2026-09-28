import json

import numpy as np
import pytest

from pipeline.build import ATTRIBUTION, build
from pipeline.encode import encode
from pipeline.grid import Source
from tests.conftest import make_tif, tile


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


def test_half_pixel_origin_lands_correctly(tmp_path):
    # EOG's rasters are pixel-centred: the corner sits half a pixel off the
    # whole degree. Construction must not raise, and align()'s floor+1 rule
    # (take the pixel centred inside the cell, east/south of the edge)
    # shifts the lit cell by exactly one row and one column.
    res = 1 / 120
    west = -122.0 - res / 2
    north = 46.0 + res / 2
    arr = np.zeros((240, 240), np.float32)
    arr[10, 20] = 10.0
    tif = tmp_path / "half.tif"
    make_tif(tif, west, north, res, arr)
    src = Source(str(tif))   # must not raise
    out = tmp_path / "site"
    m, _ = build(src, str(out), 2024, "half.tif", 50, 40, log=lambda *_: None)
    assert m["fine"]["keys"] == ["N44W122"]
    fine = tile(out / "fine" / "N44W122.bin")
    assert fine[9, 19] == 113 and int(fine.sum()) == 113


def test_misaligned_origin_raises(tmp_path):
    res = 1 / 120
    arr = np.zeros((10, 10), np.float32)
    tif = tmp_path / "bad.tif"
    make_tif(tif, -122.0 - 0.3 * res, 46.0, res, arr)
    with pytest.raises(ValueError):
        Source(str(tif))


def test_band_continuity_no_duplicated_rows(tmp_path):
    # A half-pixel origin spanning two fine bands (48..46, 46..44): under the
    # old per-band float rounding, adjacent bands could disagree by one row
    # (duplicating or dropping a source row at the seam). The two lit source
    # rows here are adjacent and must each land exactly once, in the two
    # neighbouring tiles.
    res = 1 / 120
    west = -122.0 - res / 2
    north = 48.0 + res / 2
    arr = np.zeros((480, 480), np.float32)   # ~lat 48..44, lon -122..-118
    arr[240, 10] = 5.0   # last source row read into the 48..46 fine band
    arr[241, 10] = 7.0   # first source row read into the 46..44 fine band
    tif = tmp_path / "cont.tif"
    make_tif(tif, west, north, res, arr)
    out = tmp_path / "site"
    m, _ = build(Source(str(tif)), str(out), 2024, "cont.tif", 50, 40, log=lambda *_: None)
    assert sorted(m["fine"]["keys"]) == ["N44W122", "N46W122"]
    upper = tile(out / "fine" / "N46W122.bin")
    lower = tile(out / "fine" / "N44W122.bin")
    e5, e7 = int(encode(5.0)), int(encode(7.0))
    assert upper[239, 9] == e5 and int(upper.sum()) == e5
    assert lower[0, 9] == e7 and int(lower.sum()) == e7


def test_nan_pixels_are_dark(tmp_path):
    arr = np.zeros((240, 240), np.float32)
    arr[5, 5] = np.nan
    arr[6, 6] = 8.0
    tif = tmp_path / "nan.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    out = tmp_path / "site"
    build(Source(str(tif)), str(out), 2024, "nan.tif", 50, 40, log=lambda *_: None)
    fine = tile(out / "fine" / "N44W122.bin")
    assert fine[5, 5] == 0
    assert fine[6, 6] == int(encode(8.0))


def test_nonempty_out_dir_raises(tmp_path):
    arr = np.zeros((240, 240), np.float32)
    arr[0, 0] = 5.0
    tif = tmp_path / "p.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.txt").write_text("leftover")
    with pytest.raises(ValueError):
        build(Source(str(tif)), str(out), 2024, "p.tif", 50, 40, log=lambda *_: None)
