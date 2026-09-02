#!/usr/bin/env python
"""Import the offline bundle on the HF-blocked workstation.

gh release download → verify every archive checksum against the manifest →
unpack (HF cache layout + corpus shards + dedup-hash cache) → write HF_HOME
and offline flags into the conda activation hook → run the gates:
  1. model field-check offline (scripts/m0_smoke.py, must load from the cache)
  2. v1.3 rebuild: finalize the bundled shards offline (concatenate + shuffle +
     the full dedup gate against the bundled hash cache)
  3. finalize stats must MATCH DATA.md: n_pairs 375,671 / K>=3 33.4% /
     dedup removed 17,268

Usage (from the repo root on the workstation):
    python scripts/import_offline_bundle.py [--version bundle-v1]
        [--hf-home ~/hf_home] [--gh gh] [--skip-gates]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED = {"n_pairs": 375671, "multichunk_share_pairs": 0.332, "removed_pairs": 17268}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="bundle-v1")
    ap.add_argument("--hf-home", default=str(Path.home() / "hf_home"))
    ap.add_argument("--gh", default="gh")
    ap.add_argument("--download-dir", default="/tmp/abstractnet_bundle_in")
    ap.add_argument("--skip-gates", action="store_true")
    args = ap.parse_args()
    dl = Path(args.download_dir)
    dl.mkdir(parents=True, exist_ok=True)

    print(f"[download] release {args.version} …")
    subprocess.run([args.gh, "release", "download", args.version, "--dir", str(dl),
                    "--clobber"], check=True)
    manifest = json.loads((dl / f"{args.version}.manifest.json").read_text())
    for a in manifest["archives"]:
        got = sha256(dl / a["name"])
        assert got == a["sha256"], f"checksum mismatch on {a['name']}"
        print(f"[verify] {a['name']} OK ({a['bytes'] / 1e9:.2f} GB)")

    stage = dl / "unpacked"
    stage.mkdir(exist_ok=True)
    parts = sorted(dl.glob(f"{args.version}.tar.zst.part-*"))
    cat = subprocess.Popen(["cat", *map(str, parts)], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-I", "zstd -d", "-xf", "-", "-C", str(stage)],
                   stdin=cat.stdout, check=True)
    assert cat.wait() == 0

    hf_home = Path(args.hf_home)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    for model_dir in sorted((stage / "hf_cache" / "hub").iterdir()):
        dst = hf_home / "hub" / model_dir.name
        if not dst.exists():
            model_dir.rename(dst)
        print(f"[hf] {model_dir.name}")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    src = stage / "data" / "processed" / "pilot_v1.3"
    if not Path("data/processed/pilot_v1.3").exists():
        src.rename("data/processed/pilot_v1.3")
    Path("runs/m1").mkdir(parents=True, exist_ok=True)
    if not Path("runs/m1/eval_hashes.txt").exists():
        (stage / "runs" / "m1" / "eval_hashes.txt").rename("runs/m1/eval_hashes.txt")

    # offline env: conda activation hook
    env_dir = Path(sys.prefix) / "etc" / "conda" / "activate.d"
    env_dir.mkdir(parents=True, exist_ok=True)
    hook = env_dir / "hf_offline.sh"
    hook.write_text(f"export HF_HOME={hf_home}\n"
                    "export HF_HUB_OFFLINE=1\nexport HF_DATASETS_OFFLINE=1\n")
    os.environ.update(HF_HOME=str(hf_home), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    print(f"[env] wrote {hook} (re-activate the env to pick it up)")

    if args.skip_gates:
        print("IMPORT COMPLETE (gates skipped)")
        return

    print("[gate 1] offline model field-check …")
    subprocess.run([sys.executable, "scripts/m0_smoke.py"], check=True, env=os.environ)
    print("[gate 2] offline v1.3 finalize (rebuild `full` + dedup gate) …")
    subprocess.run([sys.executable, "scripts/build_pilot.py",
                    "--out", "data/processed/pilot_v1.3", "--finalize"],
                   check=True, env=os.environ)
    stats = json.loads(Path("runs/m1/pilot_v1.3_stats.json").read_text())
    got = {"n_pairs": stats["n_pairs"],
           "multichunk_share_pairs": stats["multichunk_share_pairs"],
           "removed_pairs": stats["dedup"]["removed_pairs"]}
    assert got == EXPECTED, f"DATA.md mismatch: {got} != {EXPECTED}"
    print(f"[gate 3] DATA.md match OK: {got}")
    print(f"bundle git commit {manifest['git_commit'][:8]} vs local "
          f"{subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout[:8]}")
    print("IMPORT COMPLETE — all gates passed")


if __name__ == "__main__":
    main()
