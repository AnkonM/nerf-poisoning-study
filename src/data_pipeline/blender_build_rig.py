"""Generate the Phase 4 camera rig. Runs INSIDE Blender.

Produces three DISJOINT pose pools (train / val / eval_holdout, per D-021) and
writes them as both `data/blender_scenes/cameras.json` (this project's own
record, including the V_target threshold per METHODOLOGY.md §2) and the three
`transforms_{train,val,test}.json` files the vendored NeRF loader requires.

Why this runs in Blender rather than in numpy: `transform_matrix` must be
exactly the camera-to-world convention the loader expects. Phase 2 established
(and `src/nerf/datasets/blender.py` records) that this is Blender's OWN native
camera convention — camera looks down local -Z, +Y up, +X right — so taking
`camera.matrix_world` verbatim is correct BY CONSTRUCTION. Re-deriving the same
matrix in numpy would just be an opportunity to get it subtly wrong.

Also measures the actual scene depth range via Cycles' Z pass, so the config's
`near`/`far` are set from the real scene rather than inherited from Lego.

Usage (via scripts/build_camera_rig.py, not directly):
    blender.exe --background <scene.blend> --python blender_build_rig.py \
        -- <spec.json> <out_dir> <git_commit> [--measure-depth]
"""

import json
import math
import os
import random
import sys

import bpy

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def generate_positions(n, radius, el_min_deg, el_max_deg, look_at):
    """Fibonacci lattice over a spherical BAND between two elevations.

    Sampling sin(elevation) uniformly (rather than elevation itself) makes the
    poses equal-area over the band, so the rig does not over-sample the pole.
    The golden-angle azimuth gives a low-discrepancy spread with no clumping
    and no repeated azimuths.
    """
    z_min, z_max = math.sin(math.radians(el_min_deg)), math.sin(math.radians(el_max_deg))
    out = []
    for i in range(n):
        z = z_min + (i + 0.5) / n * (z_max - z_min)
        el = math.asin(z)
        az = (i * GOLDEN_ANGLE) % (2.0 * math.pi)
        out.append({
            "azimuth_deg": math.degrees(az),
            "elevation_deg": math.degrees(el),
            "location": [
                look_at[0] + radius * math.cos(el) * math.cos(az),
                look_at[1] + radius * math.cos(el) * math.sin(az),
                look_at[2] + radius * math.sin(el),
            ],
        })
    return out


def measure_depth_range(scene, cam, poses, sample_every):
    """Min/max scene depth over a sample of poses, from Cycles' Z pass.

    This is what `near`/`far` must bracket. Doing it by measurement rather than
    by hand-derived geometry catches the far backdrop wall, which is much
    farther from the camera than the subject is.
    """
    import numpy as np

    vl = bpy.context.view_layer
    vl.use_pass_z = True
    prev_res = (scene.render.resolution_x, scene.render.resolution_y,
                scene.cycles.samples)
    scene.render.resolution_x = scene.render.resolution_y = 128
    scene.cycles.samples = 1

    tree = bpy.data.node_groups.new("DepthProbe", "CompositorNodeTree")
    prev_tree = scene.compositing_node_group
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    go = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rl.outputs["Depth"], go.inputs["Image"])

    lo, hi = float("inf"), 0.0
    for p in poses[::sample_every]:
        cam.location = tuple(p["location"])
        bpy.context.view_layer.update()
        bpy.ops.render.render()
        rr = bpy.data.images["Render Result"]
        # Render Result pixels are not directly readable; round-trip via a file
        tmp = os.path.join(bpy.app.tempdir, "depth.exr")
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        rr.save_render(filepath=tmp)
        img = bpy.data.images.load(tmp)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(-1, 4)[:, 0]
        bpy.data.images.remove(img)
        finite = px[np.isfinite(px) & (px > 0) & (px < 1e6)]
        if finite.size:
            lo = min(lo, float(finite.min()))
            hi = max(hi, float(finite.max()))

    scene.compositing_node_group = prev_tree
    (scene.render.resolution_x, scene.render.resolution_y,
     scene.cycles.samples) = prev_res
    return lo, hi


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, out_dir, git_commit = argv[0], argv[1], argv[2]
    measure = "--measure-depth" in argv
    cfg = json.load(open(spec_path))

    cam_cfg = cfg["camera"]
    rig_cfg = cam_cfg["rig"]
    look_at = cam_cfg["look_at"]
    counts = rig_cfg["counts"]
    total = counts["train"] + counts["val"] + counts["eval_holdout"]

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cam = scene.camera
    cam.data.lens = cam_cfg["lens_mm"]
    cam.data.sensor_width = cam_cfg["sensor_mm"]
    scene.render.resolution_x, scene.render.resolution_y = cfg["render"]["resolution"]

    # aim every pose at the same look_at point via a TRACK_TO constraint, then
    # read matrix_world back out (see module docstring)
    bpy.ops.object.empty_add(location=tuple(look_at))
    pivot = bpy.context.active_object
    con = cam.constraints.new(type="TRACK_TO")
    con.target = pivot
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    poses = generate_positions(total, cam_cfg["radius"],
                               cam_cfg["elevation_min_deg"],
                               cam_cfg["elevation_max_deg"], look_at)

    depth_lo = depth_hi = None
    if measure:
        depth_lo, depth_hi = measure_depth_range(
            scene, cam, poses, rig_cfg.get("depth_probe_every", 8))
        print("RIG depth range over sampled poses: %.3f .. %.3f m"
              % (depth_lo, depth_hi))

    # Seeded partition into three disjoint pools. Disjointness is guaranteed by
    # construction (a partition of distinct lattice points), not by checking.
    order = list(range(total))
    random.Random(rig_cfg["partition_seed"]).shuffle(order)
    split_of = {}
    cursor = 0
    for split, n in (("train", counts["train"]), ("val", counts["val"]),
                     ("eval_holdout", counts["eval_holdout"])):
        for idx in order[cursor:cursor + n]:
            split_of[idx] = split
        cursor += n

    dir_of = {"train": "train", "val": "val", "eval_holdout": "eval_holdout"}
    per_split = {"train": [], "val": [], "eval_holdout": []}
    for i, p in enumerate(poses):
        split = split_of[i]
        cam.location = tuple(p["location"])
        bpy.context.view_layer.update()
        m = [list(row) for row in cam.matrix_world]
        seq = len(per_split[split])
        p.update({
            "index": i,
            "split": split,
            "file_path": "./%s/r_%03d" % (dir_of[split], seq),
            "transform_matrix": m,
        })
        per_split[split].append(p)

    camera_angle_x = cam.data.angle_x

    os.makedirs(out_dir, exist_ok=True)
    # The loader reads transforms_test.json for the held-out split, while
    # PROJECT_STRUCTURE.md names the DIRECTORY eval_holdout/. Both are honoured:
    # the file is transforms_test.json, its file_paths point into eval_holdout/.
    for split, fname in (("train", "transforms_train.json"),
                         ("val", "transforms_val.json"),
                         ("eval_holdout", "transforms_test.json")):
        # camera_angle_x must be IDENTICAL in all three: the loader takes it
        # from whichever split it happens to read last (D-021).
        payload = {
            "camera_angle_x": camera_angle_x,
            "frames": [{"file_path": p["file_path"],
                        "rotation": 0.0,
                        "transform_matrix": p["transform_matrix"]}
                       for p in per_split[split]],
        }
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(payload, f, indent=1)

    cameras = {
        "schema_version": 1,
        "scene": cfg["scene"]["name"],
        "generated_by": "scripts/build_camera_rig.py",
        "source_config": "configs/scenes/final_scene.yaml",
        "git_commit": git_commit,
        "camera": {
            "lens_mm": cam_cfg["lens_mm"],
            "sensor_mm": cam_cfg["sensor_mm"],
            "camera_angle_x": camera_angle_x,
            "resolution": cfg["render"]["resolution"],
            "convention": ("blender_native: camera-to-world 4x4, camera looks "
                           "down local -Z, +Y up, +X right (matches "
                           "src/nerf/datasets/blender.py, verified Phase 2)"),
        },
        "rig": {
            "method": "fibonacci_lattice_elevation_band",
            "radius": cam_cfg["radius"],
            "look_at": look_at,
            "elevation_deg": [cam_cfg["elevation_min_deg"],
                              cam_cfg["elevation_max_deg"]],
            "total_poses": total,
            "counts": counts,
            "partition_seed": rig_cfg["partition_seed"],
        },
        "depth_probe": ({"min_m": depth_lo, "max_m": depth_hi}
                        if measure else None),
        # V_target is a Step 5 deliverable; the THRESHOLD is locked now, before
        # the count is ever computed (METHODOLOGY.md §2, D-022).
        "v_target": {
            "min_visible_area_fraction": cfg["v_target"]["min_visible_area_fraction"],
            "count": None,
            "per_view_area_fraction": None,
        },
        "poses": poses,
    }
    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cameras, f, indent=1)

    # ---- self-checks -----------------------------------------------------
    errors = []
    locs = [tuple(round(c, 9) for c in p["location"]) for p in poses]
    if len(set(locs)) != total:
        errors.append("camera positions are not all distinct: %d unique of %d"
                      % (len(set(locs)), total))
    for split, n in counts.items():
        if len(per_split[split]) != n:
            errors.append("%s has %d poses, expected %d"
                          % (split, len(per_split[split]), n))
    overlap = (set(id(p) for p in per_split["train"])
               & set(id(p) for p in per_split["eval_holdout"]))
    if overlap:
        errors.append("train and eval_holdout share poses")
    els = [p["elevation_deg"] for p in poses]
    if min(els) < cam_cfg["elevation_min_deg"] - 1e-6 or max(els) > cam_cfg["elevation_max_deg"] + 1e-6:
        errors.append("elevation out of band: %.2f..%.2f" % (min(els), max(els)))

    print("RIG poses: %d total -> train %d / val %d / eval_holdout %d"
          % (total, counts["train"], counts["val"], counts["eval_holdout"]))
    print("RIG elevation span: %.2f .. %.2f deg" % (min(els), max(els)))
    print("RIG camera_angle_x: %.6f rad (%.2f deg)"
          % (camera_angle_x, math.degrees(camera_angle_x)))
    if errors:
        for e in errors:
            print("RIG ERROR: %s" % e)
        print("RIG RESULT: FAIL")
        sys.exit(1)
    print("RIG RESULT: PASS")


if __name__ == "__main__":
    main()
