from utils import sweep_summary as ss


def _r(cond, seed, psnr):
    return {"condition": cond, "seed": seed, "psnr": psnr, "run_id": "%s_seed%d" % (cond, seed)}


def test_outliers_flags_only_beyond_3db_of_other_seeds():
    rows = [_r("C1", 1, 40.0), _r("C1", 2, 44.5), _r("C1", 3, 44.0), _r("C2", 1, 44.0), _r("C2", 2, 45.0), _r("C2", 3, 46.0)]
    fl = ss.outliers(rows)
    assert [f["run_id"] for f in fl] == ["C1_seed1"]
    assert abs(fl[0]["outside_by_db"] - 4.0) < 1e-9


def test_condition_stats_mean_std():
    st = ss.condition_stats([_r("C1", 1, 10.0), _r("C1", 2, 12.0)])
    assert st["C1"]["mean"] == 11.0 and abs(st["C1"]["std"] - 2 ** 0.5) < 1e-9


def test_resume_retry_events():
    ev = [{"event": "start"}, {"event": "resume", "from_iteration": 150000}, {"event": "retry", "from_iteration": 5}]
    assert ss.resume_retry_events(ev) == ["resume@150000", "retry@5"]
