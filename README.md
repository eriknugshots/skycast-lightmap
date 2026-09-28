# skycast-lightmap

Satellite night lights as small tiles for SkyCast's light-pollution sky
(spec: SkyCast `docs/superpowers/specs/2026-09-27-light-pollution-design.md`).

Data: EOG Annual VNL V2, median-masked, 15″ — Earth Observation Group, Payne
Institute for Public Policy, Colorado School of Mines. CC BY 4.0.
https://eogdata.mines.edu/products/vnl/

Site: https://eriknugshots.github.io/skycast-lightmap/ — `manifest.json`,
`fine/<key>.bin` (2° tiles of 30″ cells, used within 50 km) and
`coarse/<key>.bin` (10° tiles of 2.5′ cells, 50–300 km). Each tile is 240×240
bytes, row 0 the north edge, zlib-deflated; byte = round(70·log10(1 + R/0.25)),
R in nW·cm⁻²·sr⁻¹ (`pipeline/encode.py`; the app's `src/skyglow.js` decodes it).
Keys name the SW corner (`N44W122`). A tile the manifest does not list is dark.

Build (run once per data year, locally):

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest -q
    .venv/bin/python run_build.py --src data/<file>.tif --year <year> --out site

then publish `site/` as the orphan `gh-pages` branch.
