# Control convergence probe — full eval-set PSNR (not the noisy 5-view proxy)

`curves.csv` / `train_loss.png` / `val_psnr_subset.png` come from
`scripts/export_tb_curves.py`, reading TensorBoard scalars logged *during*
training. `val_psnr_subset` there is a 5-view proxy (`i_val[:5]`, per
`src/nerf/training.py`'s periodic in-training check) — cheap, but noisy
enough that reading a plateau or a dip off it directly was actively
misleading (see D-033).

**`full_eval_set_psnr.csv` is the real metric**: the full 75-view frozen
`eval_holdout` set, run independently via `scripts/evaluate.py` against each
saved checkpoint, exactly as Phase 6's condition runs will be scored. This is
what the iteration-count decision (D-033) was actually based on, and what
should be cited if this curve is referenced again — not the TensorBoard
`val_psnr_subset` series.
