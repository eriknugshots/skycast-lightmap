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
