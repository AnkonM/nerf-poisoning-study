#!/usr/bin/env python
"""Write experiments/results/phase6_sweep_summary.md and the loss-curve contact sheet (read-only over runs)."""

import os
import sys

from utils import sweep_summary as ss

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    exp = os.path.join(REPO_ROOT, "experiments")
    rows = ss.collect_rows(os.path.join(exp, "runs"), os.path.join(exp, "results", "results.csv"))
    if len(rows) != 24:
        sys.exit("expected 24 results rows, found %d" % len(rows))
    md = os.path.join(exp, "results", "phase6_sweep_summary.md")
    with open(md, "w") as f:
        f.write(ss.render_markdown(rows))
    pngs = [os.path.join(exp, "logs", "%s_loss.png" % r["run_id"]) for r in rows]
    missing = [p for p in pngs if not os.path.isfile(p)]
    if missing:
        sys.exit("missing loss PNGs: %s" % missing)
    sheet = os.path.join(exp, "results", "phase6_loss_curves_contact_sheet.png")
    ss.contact_sheet(pngs, [r["run_id"] for r in rows], sheet)
    print("wrote", md, "and", sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
