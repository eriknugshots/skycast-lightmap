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

Build (run once per data year, locally — there is no scheduled job and no
download script; fetch the year's `*median_masked*.tif.gz` from EOG by hand,
which needs an EOG sign-in, and gunzip it into `data/`):

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest -q
    .venv/bin/python run_build.py --src data/<file>.tif --year <year> --out build-<year>

The build goes to `build-<year>.building` and is renamed to `build-<year>`
only when every check passes (manifest is the app's, every listed tile is
present and 240×240, no stray tiles, a full world has ≥ 3,900 fine / 300
coarse tiles, known bright and dark places read right). A failed build never
produces `build-<year>`, so the published year stays live.

Publish: `site/` is the `gh-pages` working copy. Replace its `fine/`,
`coarse/` and `manifest.json` with the new build's, then

    .venv/bin/python run_build.py --check site     # before the push
    (cd site && git add -A && git commit -m "site: VNL V2 <year>" && git push)
    .venv/bin/python run_build.py --check-url      # after Pages deploys

`--check-url` reads the live manifest and a sample of tiles, retrying dropped
connections, timeouts, short bodies and 5xx; a listed tile the live site lacks
fails the app's whole load for that place, so re-push until it passes. To roll
back, `git revert` the site commit and push.
