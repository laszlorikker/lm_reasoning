#!/usr/bin/env python
"""Export the offline bundle for the HF-blocked remote workstation and upload
it as GitHub Release assets (the exclusive transport).

Bundle contents (enumerated, not guessed):
  hf_cache/hub/  — snapshots of every model the code loads: the base LM
                   (config.model.name_or_path) and both NLI models in
                   eval/nli.py (primary + fallback)
  data/processed/pilot_v1.3/<shards>  — the corpus shards (NOT `full`: the
                   import gate rebuilds `full` offline via the finalize+dedup
                   path, which is the point of the gate). This pipeline has no
                   raw-text stage — with HF blocked, the shards are the rawest
                   reproducible artifact.
  runs/m1/eval_hashes.txt — the dedup gate's hash cache (offline finalize)
  wheelhouse/ (only with --with-wheelhouse) — pip wheels for the pinned env
  bundle_manifest.json — version, git commit, per-file and per-archive sha256

Archives: one zstd tar stream split into parts <= 1.9 GB (GitHub's asset limit
is 2 GB); each part uploaded with per-asset retry; the release is tagged with
the bundle manifest version.

Usage:
    python scripts/export_offline_bundle.py [--version bundle-v1] [--stage DIR]
        [--gh ~/bin/gh] [--skip-upload] [--with-wheelhouse]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from abstractnet.config import load_config

PART_BYTES = 1_900_000_000  # <= 1.9 GB per release asset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_ids() -> list[str]:
    from eval.nli import FALLBACK, PRIMARY

    cfg = load_config("configs/base.yaml")
    return [cfg.model.name_or_path, PRIMARY, FALLBACK]


def stage_bundle(stage: Path, with_wheelhouse: bool) -> list[Path]:
    from huggingface_hub import snapshot_download

    if stage.exists():
        shutil.rmtree(stage)
    (stage / "hf_cache" / "hub").mkdir(parents=True)

    for mid in model_ids():
        local = Path(snapshot_download(mid))  # cached; downloads only if absent
        model_dir = local.parents[1]  # .../hub/models--org--name (blobs+snapshots+refs)
        dst = stage / "hf_cache" / "hub" / model_dir.name
        print(f"[stage] {mid} -> {model_dir.name}")
        # keep the relative snapshot->blob symlinks: tar preserves them and the
        # HF cache layout stays valid; materialising would double the size
        shutil.copytree(model_dir, dst, symlinks=True)

    corpus_src = Path("data/processed/pilot_v1.3")
    corpus_dst = stage / "data" / "processed" / "pilot_v1.3"
    for shard in sorted(corpus_src.iterdir()):
        if shard.is_dir() and shard.name != "full":
            shutil.copytree(shard, corpus_dst / shard.name)
    (stage / "runs" / "m1").mkdir(parents=True)
    shutil.copy("runs/m1/eval_hashes.txt", stage / "runs" / "m1" / "eval_hashes.txt")

    if with_wheelhouse:
        wh = stage / "wheelhouse"
        wh.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "download", "-r", "requirements.txt",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu126",
                        "-d", str(wh)], check=True)

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="bundle-v1")
    ap.add_argument("--stage", default="/tmp/abstractnet_bundle_stage")
    ap.add_argument("--out", default="/tmp/abstractnet_bundle_out")
    ap.add_argument("--gh", default=str(Path.home() / "bin" / "gh"))
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--upload-only", action="store_true",
                    help="upload existing parts + manifest from --out; no re-staging")
    ap.add_argument("--with-wheelhouse", action="store_true")
    ap.add_argument("--zstd-threads", type=int, default=4,
                    help="politeness cap: the pilot may share this CPU")
    args = ap.parse_args()
    stage, out = Path(args.stage), Path(args.out)

    if args.upload_only:
        manifest_path = out / f"{args.version}.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        parts = sorted(out.glob(f"{args.version}.tar.zst.part-*"))
        assert parts and len(parts) == len(manifest["archives"]), "parts missing vs manifest"
        for a, p in zip(manifest["archives"], parts):
            assert p.name == a["name"] and sha256(p) == a["sha256"], f"stale part {p.name}"
        git_commit = manifest["git_commit"]
        upload(args, manifest, manifest_path, parts, git_commit)
        return

    files = stage_bundle(stage, args.with_wheelhouse)
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    manifest = {
        "version": args.version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit,
        "models": model_ids(),
        "files": [{"path": str(p.relative_to(stage)), "bytes": p.stat().st_size,
                   "sha256": sha256(p)} for p in files],
    }

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{args.version}*"):
        old.unlink()
    prefix = out / f"{args.version}.tar.zst.part-"
    print(f"[tar] {sum(f['bytes'] for f in manifest['files']) / 1e9:.2f} GB staged; "
          f"compressing + splitting at {PART_BYTES / 1e9:.1f} GB…")
    tar = subprocess.Popen(
        ["tar", "-I", f"zstd -T{args.zstd_threads} -6", "-cf", "-", "-C", str(stage), "."],
        stdout=subprocess.PIPE)
    subprocess.run(["split", "-b", str(PART_BYTES), "-d", "-a", "2", "-",
                    str(prefix)], stdin=tar.stdout, check=True)
    assert tar.wait() == 0, "tar/zstd failed"

    parts = sorted(out.glob(f"{args.version}.tar.zst.part-*"))
    manifest["archives"] = [{"name": p.name, "bytes": p.stat().st_size,
                             "sha256": sha256(p)} for p in parts]
    manifest_path = out / f"{args.version}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[parts] {len(parts)} archives, "
          f"{sum(a['bytes'] for a in manifest['archives']) / 1e9:.2f} GB total")

    if args.skip_upload:
        print(f"upload skipped; assets in {out}")
        return
    upload(args, manifest, manifest_path, parts, git_commit)


def upload(args, manifest, manifest_path: Path, parts: list[Path], git_commit: str) -> None:
    gh = args.gh
    subprocess.run([gh, "release", "create", args.version, "--title",
                    f"Offline bundle {args.version}", "--notes",
                    f"git {git_commit[:8]}; models: {', '.join(manifest['models'])}; "
                    f"{len(parts)} parts. Import: scripts/import_offline_bundle.py"],
                   check=False)  # tolerates an existing tag on re-run
    for asset in [manifest_path] + parts:
        for attempt in range(1, 4):
            r = subprocess.run([gh, "release", "upload", args.version, str(asset),
                               "--clobber"])
            if r.returncode == 0:
                print(f"[upload] {asset.name} OK")
                break
            print(f"[upload] {asset.name} attempt {attempt} failed; retrying in 10s")
            time.sleep(10)
        else:
            sys.exit(f"upload failed after 3 attempts: {asset.name}")
    print(f"EXPORT COMPLETE: release {args.version}")


if __name__ == "__main__":
    main()
