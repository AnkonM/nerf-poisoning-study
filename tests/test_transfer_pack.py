import os
import zipfile

from utils import transfer_pack as T


def test_plan_parts_keeps_groups_whole_and_respects_limit():
    parts = T.plan_parts({"a": 5, "b": 4, "c": 6, "d": 20}, limit=10)
    assert parts == [["a", "b"], ["c"], ["d"]]


def test_write_zip_uses_repo_relative_paths_and_manifest(tmp_path):
    root = tmp_path / "repo"
    f = root / "experiments" / "runs" / "r1" / "x.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"hello")
    n, size = T.write_zip(str(tmp_path / "o.zip"), str(root), [str(f)])
    assert n == 1 and size > 0
    assert zipfile.ZipFile(str(tmp_path / "o.zip")).namelist() == ["experiments/runs/r1/x.bin"]
    T.write_manifest(str(tmp_path / "m"), str(root), [str(f)])
    assert open(str(tmp_path / "m")).read().split() == [T.sha256_file(str(f)), "experiments/runs/r1/x.bin"]
