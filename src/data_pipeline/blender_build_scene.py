"""Procedural build of the Phase 4 main-study scene. Runs INSIDE Blender.

This module is executed by Blender's bundled Python (`blender.exe --background
--python ...`), NOT by the project's uv venv, so it may only import `bpy` and
the standard library. In particular Blender's bundled Python has no `yaml`,
which is why it is handed an already-resolved JSON spec by
`scripts/build_scene.py` rather than reading `configs/` itself. The YAML under
`configs/` remains the single source of truth; the JSON is a derived, ephemeral
artifact.

Per the scene brief, the scene is built procedurally from primitives rather
than modeled by hand in the GUI: a hand-modeled .blend is not reproducible or
auditable from a config + commit hash, which is this project's core ground rule
(README.md). data/raw/final_scene.blend is therefore a build *artifact* of this
script, regenerable at any time.

Usage (via scripts/build_scene.py, not directly):
    blender.exe --background --factory-startup \
        --python blender_build_scene.py -- <spec.json> <out.blend>
"""

import json
import math
import sys

import bpy
import mathutils


# --------------------------------------------------------------------------
# scene teardown / helpers
# --------------------------------------------------------------------------

def _clear_scene():
    """Remove everything, including orphaned datablocks.

    --factory-startup still gives us the default cube/camera/light, and stale
    datablocks would otherwise persist into the saved .blend and make the build
    non-deterministic.
    """
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.lights,
                       bpy.data.cameras, bpy.data.objects):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _make_material(name, color, roughness):
    """Opaque, diffuse Principled BSDF.

    Explicitly zero metallic and zero transmission: the scene brief forbids
    glass/mirror/metal because specular and refractive materials make the
    soft-suppression blend ambiguous and break object-ID mask cleanliness.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Transmission Weight" in bsdf.inputs:          # Blender 4.x+ naming
        bsdf.inputs["Transmission Weight"].default_value = 0.0
    elif "Transmission" in bsdf.inputs:               # older naming
        bsdf.inputs["Transmission"].default_value = 0.0
    return mat


def _assign(obj, mat, pass_index, smooth_angle_deg=30.0):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    # pass_index drives the compositor ID Mask node that produces the mask.
    # Only the target is non-zero.
    obj.pass_index = pass_index
    # Auto-smooth by angle, NOT unconditional shade_smooth: blanket smooth
    # shading rounds off the cone tip, the cylinder caps and the box edges so
    # they read as melted blobs. Auto-smooth keeps sharp edges sharp while
    # smoothing the genuinely curved surfaces.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_auto_smooth(angle=math.radians(smooth_angle_deg))


def _ring_xy(angle_deg, radius):
    a = math.radians(angle_deg)
    return radius * math.cos(a), radius * math.sin(a)


# --------------------------------------------------------------------------
# object builders
# --------------------------------------------------------------------------

def _build_target(spec):
    """Mug = tapered cylinder body + torus handle, joined into one object.

    Joined deliberately: the mask is keyed on a single pass_index, so body and
    handle must be one object or the handle would be excluded from the mask.
    """
    body = spec["body"]
    handle = spec["handle"]
    loc = spec["location"]

    bpy.ops.mesh.primitive_cone_add(
        radius1=body["radius_bottom"],
        radius2=body["radius_top"],
        depth=body["height"],
        vertices=body["vertices"],
        location=(loc[0], loc[1], loc[2] + body["height"] / 2.0),
    )
    body_obj = bpy.context.active_object
    body_obj.name = spec["name"]

    off = handle["offset"]
    bpy.ops.mesh.primitive_torus_add(
        major_radius=handle["major_radius"],
        minor_radius=handle["minor_radius"],
        major_segments=handle["major_segments"],
        minor_segments=handle["minor_segments"],
        location=(loc[0] + off[0], loc[1] + off[1], loc[2] + off[2]),
        # stand the torus up in the XZ plane so it reads as a side handle
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    handle_obj = bpy.context.active_object

    bpy.ops.object.select_all(action="DESELECT")
    handle_obj.select_set(True)
    body_obj.select_set(True)
    bpy.context.view_layer.objects.active = body_obj
    bpy.ops.object.join()

    target = bpy.context.active_object
    target.name = spec["name"]
    _assign(target, _make_material("M_" + spec["name"], spec["color"],
                                   spec["roughness"]), spec["pass_index"])
    return target


def _build_distractor(spec):
    x, y = _ring_xy(spec["ring_angle_deg"], spec["ring_radius"])
    kind = spec["type"]

    if kind == "box":
        d = spec["dimensions"]
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, d[2] / 2.0))
        obj = bpy.context.active_object
        obj.scale = (d[0], d[1], d[2])
        obj.rotation_euler[2] = math.radians(spec.get("rotation_z_deg", 0.0))
    elif kind == "sphere":
        r = spec["radius"]
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(x, y, r))
        obj = bpy.context.active_object
    elif kind == "cone":
        h = spec["height"]
        bpy.ops.mesh.primitive_cone_add(radius1=spec["radius"], radius2=0.0,
                                        depth=h, vertices=spec["vertices"],
                                        location=(x, y, h / 2.0))
        obj = bpy.context.active_object
    elif kind == "cylinder":
        h = spec["height"]
        bpy.ops.mesh.primitive_cylinder_add(radius=spec["radius"], depth=h,
                                            vertices=spec["vertices"],
                                            location=(x, y, h / 2.0))
        obj = bpy.context.active_object
    elif kind == "torus":
        # lies flat in the XY plane, so it rests on the table at z = minor
        bpy.ops.mesh.primitive_torus_add(
            major_radius=spec["major_radius"], minor_radius=spec["minor_radius"],
            major_segments=spec["major_segments"],
            minor_segments=spec["minor_segments"],
            location=(x, y, spec["minor_radius"]))
        obj = bpy.context.active_object
    else:
        raise ValueError("unknown distractor type: %r" % kind)

    obj.name = spec["name"]
    _assign(obj, _make_material("M_" + spec["name"], spec["color"],
                                spec["roughness"]), 0)
    return obj


def _build_environment(scene_spec):
    t = scene_spec["table"]
    bpy.ops.mesh.primitive_circle_add(radius=t["radius"], vertices=128,
                                      fill_type="NGON", location=(0, 0, 0))
    table = bpy.context.active_object
    table.name = t["name"]
    _assign(table, _make_material("M_Table", t["color"], t["roughness"]), 0)

    # Open-topped cylinder ("cyclorama"): guarantees no 360-degree rig azimuth
    # can see past the edge of the world, which a flat backdrop would allow.
    b = scene_spec["backdrop"]
    bpy.ops.mesh.primitive_cylinder_add(radius=b["radius"], depth=b["height"],
                                        vertices=128,
                                        location=(0, 0, b["height"] / 2.0))
    backdrop = bpy.context.active_object
    backdrop.name = b["name"]
    # delete the caps so we have a wall, not a sealed drum
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for poly in backdrop.data.polygons:
        if abs(poly.normal.z) > 0.9:
            poly.select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    # inward-facing normals so the wall is lit from inside
    bpy.ops.object.select_all(action="DESELECT")
    backdrop.select_set(True)
    bpy.context.view_layer.objects.active = backdrop
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    _assign(backdrop, _make_material("M_Backdrop", b["color"], b["roughness"]), 0)


def _build_lighting(light_spec):
    k = light_spec["key_light"]
    bpy.ops.object.light_add(type="AREA", location=tuple(k["location"]))
    light = bpy.context.active_object
    light.name = k["name"]
    light.data.size = k["size"]
    light.data.energy = k["energy"]
    # Aim the key light at the origin. Use to_track_quat rather than hand-rolled
    # Euler math: an earlier hand-rolled version had a sign error (acos(dz/r)
    # instead of acos(-dz/r)) that aimed the light UPWARD at the backdrop. The
    # scene still rendered -- lit entirely by world ambient and backdrop bounce
    # -- so it failed silently as "flat, shadowless lighting" rather than as an
    # error, and barely responded to changes in the light's energy.
    aim = (mathutils.Vector((0.0, 0.0, 0.0))
           - mathutils.Vector(tuple(k["location"])))
    light.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()

    w = light_spec["world_background"]
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (w["color"][0], w["color"][1],
                                        w["color"][2], 1.0)
    bg.inputs["Strength"].default_value = w["strength"]


def _build_camera(cam_spec):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.active_object
    cam.name = "Camera"
    cam.data.lens = cam_spec["lens_mm"]
    cam.data.sensor_width = cam_spec["sensor_mm"]
    bpy.context.scene.camera = cam
    return cam


def _configure_render(render_spec):
    scene = bpy.context.scene
    scene.render.engine = render_spec["engine"]
    scene.render.resolution_x = render_spec["resolution"][0]
    scene.render.resolution_y = render_spec["resolution"][1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = render_spec["file_format"]
    scene.render.image_settings.color_mode = render_spec["color_mode"]
    # Linear-sRGB, not a photographic tone-map. Config-driven rather than left
    # in a render script, since it changes every pixel of every training image.
    scene.view_settings.view_transform = render_spec["view_transform"]

    scene.cycles.device = render_spec["device"]
    scene.cycles.samples = render_spec["samples"]
    scene.cycles.use_denoising = render_spec["use_denoising"]
    # Determinism: a fixed seed that does NOT advance per frame is what makes
    # the original and background_plate renders of a view comparable.
    scene.cycles.seed = render_spec["seed"]
    scene.cycles.use_animated_seed = False

    # Object Index pass -> compositor ID Mask node is how the mask is produced
    # as an extra output of the SAME render as the original image, rather than
    # a third path-traced pass.
    bpy.context.view_layer.use_pass_object_index = True


# --------------------------------------------------------------------------

def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec_path, out_blend = argv[0], argv[1]
    with open(spec_path, "r") as f:
        cfg = json.load(f)

    scene_spec = cfg["scene"]

    _clear_scene()
    _build_environment(scene_spec)
    target = _build_target(scene_spec["target"])
    distractors = [_build_distractor(d) for d in scene_spec["distractors"]]
    _build_lighting(scene_spec["lighting"])
    _build_camera(cfg["camera"])
    _configure_render(cfg["render"])

    # ---- self-verification (Step 1 PASS/FAIL, mechanical) ----------------
    errors = []
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    expected_meshes = 2 + 1 + len(scene_spec["distractors"])   # table+backdrop+target+distractors
    if len(meshes) != expected_meshes:
        errors.append("expected %d mesh objects, found %d: %s"
                      % (expected_meshes, len(meshes), [o.name for o in meshes]))
    if target.pass_index != 1:
        errors.append("target pass_index is %d, expected 1" % target.pass_index)
    for o in meshes:
        if o is not target and o.pass_index != 0:
            errors.append("%s has pass_index %d, expected 0" % (o.name, o.pass_index))
    if bpy.context.scene.camera is None:
        errors.append("no active camera")

    # target silhouette half-width, used to sanity-check distractor clearance
    tmin = min((target.matrix_world @ v.co).xy.length for v in target.data.vertices)
    tmax = max((target.matrix_world @ v.co).xy.length for v in target.data.vertices)
    for d in distractors:
        dmin = min((d.matrix_world @ v.co).xy.length for v in d.data.vertices)
        if dmin <= tmax:
            errors.append("%s (nearest %.3f m) overlaps target extent (%.3f m)"
                          % (d.name, dmin, tmax))

    print("BUILD target radial extent: %.3f - %.3f m" % (tmin, tmax))
    for o in meshes:
        print("BUILD object: %-20s pass_index=%d" % (o.name, o.pass_index))

    if errors:
        for e in errors:
            print("BUILD ERROR: %s" % e)
        print("BUILD RESULT: FAIL")
        sys.exit(1)

    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("BUILD saved: %s" % out_blend)
    print("BUILD RESULT: PASS")


if __name__ == "__main__":
    main()
