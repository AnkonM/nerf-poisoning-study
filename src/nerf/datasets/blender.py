"""Loader for the Blender Synthetic dataset format (NeRF paper, Lego etc.).

Vendored/adapted from yenchenlin/nerf-pytorch's load_blender.py (MIT
license, see src/nerf/THIRD_PARTY_LICENSE), commit
63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd. Camera-pose convention (see
this function's docstring) is the standard NeRF-synthetic one: right-
handed world coordinates, per-frame `transform_matrix` is a 4x4
camera-to-world matrix, camera looks down its own local -Z axis with +Y
up and +X right (OpenGL/Blender camera convention). This is recorded here
so it can be cross-checked against the custom Phase 4 scene's convention
per docs/ROADMAP.md Phase 2's explicit note.

Phase 5 addition (DECISION_LOG.md D-028): optional per-split base directories.
The upstream loader resolves all three splits under one `basedir`. Poisoning
needs the *train* split to come from a per-condition directory while `val` and
the frozen `eval_holdout` are read from their single canonical location — never
copied or symlinked per condition, which is what the Phase 4 freeze exists to
guarantee. The new arguments are optional and default to `basedir`, so callers
that do not pass them (configs/scenes/lego_sanity.yaml, the Phase 3 PoC configs)
get byte-identical behaviour to before.
"""

import json
import os

import cv2
import imageio
import numpy as np


def load_blender_data(basedir: str, half_res: bool = False, testskip: int = 1,
                      val_dir: str = None, test_dir: str = None):
    """Load train/val/test splits for one Blender-synthetic scene.

    Parameters
    ----------
    basedir : directory holding `transforms_train.json` (and, unless overridden
        below, the val/test transforms too). Frame `file_path`s are resolved
        relative to the base directory of their own split.
    val_dir, test_dir : optional base directories for the val and test splits.
        `None` means "same as basedir" — the upstream behaviour.

    Returns (imgs [N,H,W,4] RGBA float32 in [0,1], poses [N,4,4] float32
    camera-to-world matrices, hwf [H, W, focal], i_split [i_train, i_val,
    i_test] arrays of indices into imgs/poses).
    """
    splits = ["train", "val", "test"]
    bases = {"train": basedir,
             "val": val_dir if val_dir else basedir,
             "test": test_dir if test_dir else basedir}

    metas = {}
    for s in splits:
        with open(os.path.join(bases[s], f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    # Focal length is derived from camera_angle_x. Upstream took it from
    # whichever split happened to be parsed last, which is only safe because the
    # three files agree. With per-split directories they could come from
    # different scenes entirely, so disagreement is now checked rather than
    # assumed — a silent mismatch would yield a wrong focal length and quietly
    # wrong geometry for every downstream metric.
    angles = {s: float(metas[s]["camera_angle_x"]) for s in splits}
    if len(set(angles.values())) != 1:
        raise ValueError(
            "camera_angle_x differs between splits %r — the split directories "
            "do not describe the same camera: %s"
            % (angles, {s: bases[s] for s in splits}))
    camera_angle_x = angles["train"]

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        skip = 1 if (s == "train" or testskip == 0) else testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(bases[s], frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep RGBA
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, [H, W, focal], i_split
