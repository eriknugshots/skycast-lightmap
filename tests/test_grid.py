import numpy as np
import pytest

from pipeline.grid import Source, align
from tests.conftest import make_tif


def test_align_whole_pixel():
    assert align(1000.0, "x") == 1000
    assert align(1000.0 + 1e-7, "x") == 1000


def test_align_half_pixel():
    assert align(999.5, "x") == 1000
    assert align(-6959.5, "x") == -6959   # floor(-6959.5) + 1


def test_align_off_grid_raises():
    with pytest.raises(ValueError):
        align(1000.3, "x")


def test_non_float32_source_raises(tmp_path):
    arr = np.zeros((10, 10), np.uint16)
    tif = tmp_path / "bad_dtype.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr, dtype="uint16")
    with pytest.raises(ValueError):
        Source(str(tif))


def test_multi_band_source_raises(tmp_path):
    arr = np.zeros((2, 10, 10), np.float32)
    tif = tmp_path / "bad_bands.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr, count=2)
    with pytest.raises(ValueError):
        Source(str(tif))


def test_eog_rounded_pixel_size_still_aligns(tmp_path):
    # EOG's 2025 VNL header stores the pixel size to 10 digits (0.0041666667,
    # not 1/240) on a pixel-centred origin (-180.00208333335, 75.00208333335).
    # Dividing by the rounded size drifts 2.9e-5 px over the 3,600 rows up to
    # lat 90 and failed align(); offsets use the exact 1/per_deg instead.
    arr = np.zeros((480, 480), np.float32)
    make_tif(tmp_path / "eog.tif", -122.00208333335, 75.00208333335, 0.0041666667, arr)
    s = Source(str(tmp_path / "eog.tif"))
    assert s.per_deg == 240
    assert s.row90 == -3599 and s.col180 == align((-180.0 + 122.00208333335) * 240, "lon -180")
