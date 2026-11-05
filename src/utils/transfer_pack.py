"""Packaging for manual transfer (Phase 6 closeout, D-041): split-by-run zips < 3.5 GB, manifest, checkpoint CSV."""

import csv
import hashlib
import os
import zipfile
from typing import Dict, List, Tuple

LIMIT_BYTES = int(3.5 * 1000 ** 3)          # FAT32 max file is 4 GiB; stay well below 3.5 GB (decimal)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def list_files(root: str) -> List[str]:
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.join(d, f))
    return sorted(out)


def plan_parts(group_sizes: Dict[str, int], limit: int = LIMIT_BYTES) -> List[List[str]]:
    """Greedy, in order, whole groups (runs) per part. A group over `limit` gets its own part (caller reports it)."""
    parts: List[List[str]] = []
    cur: List[str] = []
    cur_size = 0
    for g, s in group_sizes.items():
        if cur and cur_size + s > limit:
            parts.append(cur)
            cur, cur_size = [], 0
        cur.append(g)
        cur_size += s
    if cur:
        parts.append(cur)
    return parts


def write_zip(zip_path: str, repo_root: str, files: List[str]) -> Tuple[int, int]:
    """Write files (absolute) with repo-relative names. STORED: checkpoints/fp16 do not compress. Returns (n, bytes)."""
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            z.write(p, os.path.relpath(p, repo_root))
    os.rename(tmp, zip_path)
    return len(files), os.path.getsize(zip_path)


def write_manifest(path: str, repo_root: str, files: List[str]) -> None:
    with open(path, "w") as f:
        for p in files:
            f.write("%s  %s\n" % (sha256_file(p), os.path.relpath(p, repo_root)))


def write_checkpoint_csv(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "checkpoint", "sha256", "bytes"])
        w.writeheader()
        w.writerows(rows)
