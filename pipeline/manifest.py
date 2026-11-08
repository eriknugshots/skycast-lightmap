# pipeline/manifest.py — manifest.json is read by the app on every load, so it
# is replaced whole: written beside it, then renamed over it. A crash mid-write
# leaves the old manifest (and no .tmp), never a truncated one.
import json
import os


def write_manifest(out_dir, manifest):
    path = os.path.join(out_dir, "manifest.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(manifest, f, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
