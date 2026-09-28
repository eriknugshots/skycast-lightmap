import zlib

import numpy as np

from pipeline.encode import decode, encode, pack
from pipeline.grid import FINE_PER_COARSE, block_mean, tile_key


def test_goldens_match_the_app():
    # tests/skyglow.test.js in SkyCast asserts the same bytes.
    assert encode([0, 0.1, 0.25, 1, 10, 100, 1000, 5000]).tolist() == [0, 10, 21, 49, 113, 182, 252, 255]


def test_negative_nan_and_inf_are_unlit():
    assert encode([-3.0, np.nan, np.inf, -np.inf]).tolist() == [0, 0, 0, 0]


def test_round_trip_within_half_a_step():
    for r in [0.3, 2, 17, 150, 900]:
        assert abs(decode(encode(r)) / r - 1) < 0.017


def test_pack_is_zlib():
    b = (np.arange(240 * 240) % 256).astype(np.uint8)
    assert zlib.decompress(pack(b)) == b.tobytes()


def test_tile_key_is_the_apps():
    assert tile_key(44, -122) == "N44W122"
    assert tile_key(-2, 178) == "S02E178"
    assert tile_key(0, -180) == "N00W180"
    assert tile_key(-40, 10) == "S40E010"


def test_block_mean_and_ratio():
    assert FINE_PER_COARSE == 5
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert block_mean(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert block_mean(a, 1).tolist() == a.tolist()
