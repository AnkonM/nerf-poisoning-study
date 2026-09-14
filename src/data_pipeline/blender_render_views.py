"""Batch-render the Phase 4 dataset. Runs INSIDE Blender.

For every pose in cameras.json, renders per the scheme fixed in the scene brief
— TWO real Cycles renders per view, not three:

  render 1: target camera-visible -> `original` (RGBA PNG) AND `mask`, the
            latter as an extra File Output from the SAME render via the View
            Layer's Object Index pass -> ID Mask node. The mask is therefore
            free; it is not a third path-traced pass.
  render 2: target `hide_render = True` -> `background_plate` (D-022).

Masks and plates are produced for train + eval_holdout (the held-out split
needs them for METHODOLOGY.md §6's masked metrics and the secondary
plate-distance metric). The val split is convergence-monitoring only, so it
gets `original` alone.

Determinism: `cycles.seed` is fixed and `use_animated_seed` is off, so a view's
original and plate renders share sampling. D-024 verified this holds — no
denoiser bleed beyond 12/255 in the far field.

Poses are read from cameras.json and applied as `matrix_world` DIRECTLY, rather
than recomputed from azimuth/elevation. The pose files are the single source of
truth for camera placement, so the rendered images cannot drift from the
transforms the loader will read.

Usage (via scripts/render_scene.py, not directly):
    blender.exe --background <scene.blend> --python blender_render_views.py \
        -- <spec.json> <cameras.json> <data_root> [--limit N] [--splits a,b]
"""

import json
import os
import sys
import time

import bpy
import mathutils


def setup_devices():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    names = []
    for d in prefs.devices:
        # Filter on OPTIX specifically (D-021): this machine also exposes an
        # AMD iGPU under HIP and the same GPU again under CUDA.
        d.use = (d.type == "OPTIX")
        if d.use:
            names.append(d.name)
    print("RENDER optix devices: %s" % ", ".join(names))
    if not names:
        print("RENDER ERROR: no OptiX device enabled — refusing to fall back to CPU")
        sys.exit(1)


def setup_compositor(scene, target_pass_index, mask_depth):
    """Object Index pass -> ID Mask -> File Output, as a second output of the
    same render.

    Blender 5.2 API notes (each verified by introspection, each a break from
    4.x): the compositor is `scene.compositing_node_group`, not
    `scene.node_tree`; `CompositorNodeComposite` no longer exists so the tree
    ends in `NodeGroupOutput`; the socket is "Object Index" (not "IndexOB") and
    only exists under CYCLES; ID Mask's index/anti-alias are INPUT SOCKETS, not
    properties; and File Output's `format.media_type` defaults to
    MULTI_LAYER_IMAGE which hard-locks the format to OPEN_EXR_MULTILAYER until
    set to "IMAGE".
    """
    bpy.context.view_layer.use_pass_object_index = True
    tree = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    go = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rl.outputs["Image"], go.inputs["Image"])

    idm = tree.nodes.new("CompositorNodeIDMask")
    idm.inputs["Index"].default_value = target_pass_index
    idm.inputs["Anti-Alias"].default_value = False   # keep the mask exactly 0/1

    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.format.media_type = "IMAGE"
    fout.format.file_format = "PNG"
    fout.format.color_mode = "BW"
    fout.format.color_depth = mask_depth
    # save_as_render off: otherwise the view transform is applied and a 1.0 mask
    # value would not come back as full-scale.
    fout.save_as_render = False
    # The item's NAME is appended to file_name in the output filename, so the
    # item is renamed per view and file_name left empty — otherwise masks land
    # as "r_000mask.png" and no longer pair cleanly with their originals.
    item = fout.file_output_items.new("FLOAT", "mask")
    item.save_as_render = False
    fout.file_name = ""
    tree.links.new(rl.outputs["Object Index"], idm.inputs["ID value"])
    tree.links.new(idm.outputs["Alpha"], fout.inputs["mask"])
    return fout


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, cameras_path, data_root = argv[0], argv[1], argv[2]
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    only = (argv[argv.index("--splits") + 1].split(",")
            if "--splits" in argv else None)

    cfg = json.load(open(spec_path))
    cams = json.load(open(cameras_path))
    rcfg = cfg["render"]

    scene = bpy.context.scene
    setup_devices()
    scene.render.engine = rcfg["engine"]
    scene.cycles.device = rcfg["device"]
    scene.cycles.samples = rcfg["samples"]
    scene.cycles.use_denoising = rcfg["use_denoising"]
    scene.cycles.seed = rcfg["seed"]
    scene.cycles.use_animated_seed = False
    scene.render.resolution_x, scene.render.resolution_y = rcfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = rcfg["file_format"]
    scene.render.image_settings.color_mode = rcfg["color_mode"]
    scene.view_settings.view_transform = rcfg["view_transform"]

    target = bpy.data.objects[cfg["scene"]["target"]["name"]]
    fout = setup_compositor(scene, cfg["scene"]["target"]["pass_index"],
                            rcfg["mask_color_depth"])
    mask_item = fout.file_output_items[0]

    cam = scene.camera
    for c in list(cam.constraints):        # poses are absolute; no TRACK_TO
        cam.constraints.remove(c)

    # val is monitoring-only -> no mask, no plate (D-021)
    needs_aux = {"train": True, "val": False, "eval_holdout": True}
    dirs = {"train": "blender_scenes/train", "val": "blender_scenes/val",
            "eval_holdout": "blender_scenes/eval_holdout"}

    poses = [p for p in cams["poses"] if not only or p["split"] in only]
    if limit:
        poses = poses[:limit]
    total_renders = sum(2 if needs_aux[p["split"]] else 1 for p in poses)

    for d in set(dirs.values()):
        os.makedirs(os.path.join(data_root, d), exist_ok=True)
    for d in ("masks", "background_plates"):
        for s in ("train", "eval_holdout"):
            os.makedirs(os.path.join(data_root, d, s), exist_ok=True)

    print("RENDER %d poses -> %d renders (%s)"
          % (len(poses), total_renders, rcfg["resolution"]))
    done = 0
    t_start = time.time()
    for p in poses:
        split = p["split"]
        name = os.path.basename(p["file_path"])          # r_000
        aux = needs_aux[split]

        cam.matrix_world = mathutils.Matrix(p["transform_matrix"])
        bpy.context.view_layer.update()

        # --- render 1: original (+ mask, from the same render) -------------
        fout.mute = not aux
        if aux:
            fout.directory = os.path.join(data_root, "masks", split)
            mask_item.name = name
        scene.render.filepath = os.path.join(data_root, dirs[split], name + ".png")
        bpy.ops.render.render(write_still=True)
        done += 1

        # --- render 2: background plate (target fully removed) -------------
        if aux:
            fout.mute = True
            target.hide_render = True
            scene.render.filepath = os.path.join(
                data_root, "background_plates", split, name + ".png")
            bpy.ops.render.render(write_still=True)
            target.hide_render = False
            done += 1

        # Progress with elapsed + ETA, flushed every view. D-016: an unattended
        # render that looks like a hang is a known failure mode on this
        # hardware, and the fix is to make slowness distinguishable from a
        # stall from the log alone.
        el = time.time() - t_start
        eta = el / done * (total_renders - done)
        print("RENDER [%4d/%4d] %s/%s  elapsed %5.1fm  eta %5.1fm"
              % (done, total_renders, split, name, el / 60.0, eta / 60.0))
        sys.stdout.flush()

    print("RENDER DONE %d renders in %.1f min" % (done, (time.time() - t_start) / 60.0))


if __name__ == "__main__":
    main()
