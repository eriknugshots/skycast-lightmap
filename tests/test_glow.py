import math

import numpy as np

from pipeline import glow
from pipeline.glow import CLASS1_R, FAR_KM, GLOW_R0, GLOW_SCALE, SKYGLOW_C, decode_glow, encode_glow, glow_band

KM_LAT, KM_LON = 110.57, 111.32
CELL = 0.05  # coarse cells keep the FFTs small; the maths is cell-size independent
W = round(360 / CELL)
BAND_ROWS = round(2 / CELL)
MARGIN = math.ceil(FAR_KM / (CELL * KM_LAT))


def kern(d):
    d = max(1.0, d)
    return d ** -2.5 * math.exp(-d / 100)


def strip(cells, band_rows=BAND_ROWS):
    """A zero strip of band + margins with {(row, col): radiance} lit."""
    a = np.zeros((band_rows + 2 * MARGIN, W))
    for (r, c), v in cells.items():
        a[r, c] = v
    return a


RESIDUAL_BOUND = 0.09  # measured 0.081 on the high-latitude fixture


def band_lat_c(top):
    return top - BAND_ROWS * CELL / 2


def test_single_cell_matches_direct_sum():
    top, lit_row, lit_col, R = 11.0, MARGIN + BAND_ROWS // 2, 3000, 40.0
    lat_c = band_lat_c(top)
    out = glow_band(strip({(lit_row, lit_col): R}), top, CELL, MARGIN)
    assert out.shape == (BAND_ROWS, W)
    row = BAND_ROWS // 2
    src_lat = top + MARGIN * CELL - (lit_row + 0.5) * CELL  # the source row's own latitude
    area = CELL * CELL * KM_LAT * KM_LON * math.cos(math.radians(src_lat))
    km_per_col = CELL * KM_LON * math.cos(math.radians(lat_c))
    for km in (0, 5, 50, 200):
        j = round(km / km_per_col)
        d = j * km_per_col
        want = SKYGLOW_C * R * area * kern(d)
        assert math.isclose(out[row, lit_col + j], want, rel_tol=1e-6)
    far = math.ceil((FAR_KM + 1) / km_per_col)
    assert out[row, lit_col + far] == 0.0
    assert out[row, lit_col - far] == 0.0


def test_wraps_dateline():
    top = 11.0
    row = MARGIN + BAND_ROWS // 2
    east = W - 2  # lon 179.9
    a = glow_band(strip({(row, east): 40.0}), top, CELL, MARGIN)
    b = glow_band(strip({(row, 3000): 40.0}), top, CELL, MARGIN)
    k = 4
    got = a[BAND_ROWS // 2, (east + k) % W]
    assert got > 0
    assert (east + k) % W < east  # really across the dateline
    assert math.isclose(got, b[BAND_ROWS // 2, 3000 + k], rel_tol=1e-9)


def test_high_latitude_against_direct():
    top = 62.0
    lat_c = band_lat_c(top)
    rng = np.random.default_rng(7)
    cells = {}
    while len(cells) < 5:
        r = MARGIN + BAND_ROWS // 2 + int(rng.integers(-8, 9))  # within ~0.4 deg of the centre
        c = 3000 + int(rng.integers(-60, 61))
        cells[(r, c)] = float(rng.uniform(5, 80))
    out = glow_band(strip(cells), top, CELL, MARGIN)
    cos_c = math.cos(math.radians(lat_c))

    def lat_of(row):  # row in the strip; centre of the cell
        return top + MARGIN * CELL - (row + 0.5) * CELL

    # Two brute-force sums, both with each source cell's own cos for its area
    # (glowAt). (1) Longitude km scale at the band-centre cos, as the FFT does:
    # must agree to round-off. (2) Longitude scale at each viewer's own cos,
    # like glowAt: the only difference left is the band-centre dx scale, which
    # drifts by cos(lat) across the band (3% at the 1 deg edge) and is
    # amplified by (2.5 + d/100) at d <= 100 km. BOUND is the residual measured
    # on this fixture (0.081), rounded up.
    def direct_sum(r, c, own_cos):
        lat = top - (r + 0.5) * CELL
        tot, near = 0.0, False
        for (sr, sc), R in cells.items():
            slat = lat_of(sr)
            cv = math.cos(math.radians(lat)) if own_cos else cos_c
            d = math.hypot((slat - lat) * KM_LAT, (sc - c) * CELL * KM_LON * cv)
            near = near or d <= 100
            if d <= FAR_KM:
                tot += SKYGLOW_C * R * CELL * CELL * KM_LAT * KM_LON * math.cos(math.radians(slat)) * kern(d)
        return tot, near

    checked, worst = 0, 0.0
    for r in range(BAND_ROWS):
        for c in range(2700, 3300):
            exact, near = direct_sum(r, c, False)
            if not near:
                continue
            assert math.isclose(out[r, c], exact, rel_tol=1e-6), (r, c)
            worst = max(worst, abs(out[r, c] / direct_sum(r, c, True)[0] - 1))
            checked += 1
    assert worst < RESIDUAL_BOUND, worst
    assert checked > 1000
    assert cos_c > 0


def test_encoding_round_trip_and_class_steps():
    for r in [0.0, 0.01, 0.2, 1.7, 30.0, 900.0]:
        b = int(encode_glow(np.array([r]))[0])
        lo, hi = decode_glow(max(b - 1, 0)), decode_glow(min(b + 1, 255))
        assert lo - 1e-12 <= r <= hi + 1e-12
        assert abs(40 * math.log10(1 + r / 0.002) - b) <= 0.5 + 1e-9
    assert encode_glow(np.array([-1.0, np.nan, np.inf])).tolist() == [0, 0, 0]
    assert encode_glow(np.array([1e9]))[0] == 255
    bounds = [21.99, 21.89, 21.69, 20.49, 19.50, 18.94, 18.38, 17.80]
    steps = [GLOW_SCALE * math.log10(1 + (10 ** (0.4 * (22.0 - m)) - 1) / GLOW_R0) for m in bounds]
    for a, b in zip(steps, steps[1:]):
        assert b - a >= 2


def test_class1_r():
    # CLASS1_R is the class-1 / class-2 edge: it must sit between the byte
    # steps of the neighbouring bounds (21.99 and 21.89) and encode to 30 steps.
    step = lambda m: GLOW_SCALE * math.log10(1 + (10 ** (0.4 * (22.0 - m)) - 1) / GLOW_R0)
    assert int(encode_glow(CLASS1_R)) == round(step(21.99))
    assert step(21.99) < step(21.89) - 2
    assert CLASS1_R < 10 ** (0.4 * (22.0 - 21.89)) - 1
    assert (GLOW_SCALE, GLOW_R0, SKYGLOW_C, FAR_KM) == (40, 0.002, 0.0576, 300)


def test_margin_too_small_is_refused():
    import pytest
    with pytest.raises(ValueError):
        glow_band(np.zeros((BAND_ROWS + 2, W)), 11.0, CELL, 1)
