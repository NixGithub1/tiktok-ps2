"""
Procedural generator for the "Solatnill" character (T-pose, rigged) -> FBX / GLB / BLEND.

Run with the Blender Python module (pip install bpy) or inside Blender:
    python tools/generate_solatnill.py --font path/to/gothic.ttf --out assets/solatnill
    blender -b -P tools/generate_solatnill.py -- --font ... --out ...

Units: metres, Z up, character faces -Y (Blender "front" view). Height ~1.80 m.
Bone names follow the Mixamo convention (Hips, Spine, LeftArm, ...) so
Mixamo / Unity Humanoid / Unreal retargeting map them automatically.
"""
import argparse
import math
import os
import random
import sys

import bpy  # noqa: must be imported before bmesh/mathutils
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

# --------------------------------------------------------------------------- args
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--font", default="")
ap.add_argument("--out", default="assets/solatnill")
ap.add_argument("--render", action="store_true", help="render preview images")
args = ap.parse_args(argv)
OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
random.seed(7)

# --------------------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
COLL = scene.collection


# --------------------------------------------------------------------------- materials
def srgb(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def make_mat(name, rgb255, rough=0.6, metal=0.0, spec=0.4):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    col = [srgb(v) for v in rgb255] + [1.0]
    m.diffuse_color = col
    m.roughness = rough
    m.metallic = metal
    bsdf = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
    if bsdf:
        bsdf.inputs["Base Color"].default_value = col
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = spec
    return m


MAT = {
    "skin": make_mat("Skin", (118, 68, 44), 0.5),
    "lips": make_mat("Lips", (88, 46, 34), 0.5),
    "eye_white": make_mat("EyeWhite", (205, 195, 185), 0.3),
    "iris": make_mat("Iris", (28, 18, 14), 0.2),
    "brow": make_mat("FacialHair", (20, 14, 12), 0.8),
    "hair": make_mat("Hair", (10, 9, 9), 0.75, spec=0.25),
    "fade": make_mat("HairFade", (34, 26, 23), 0.9, spec=0.2),
    "shirt": make_mat("Shirt", (226, 226, 222), 0.85, spec=0.2),
    "trim": make_mat("ShirtTrim", (150, 150, 148), 0.85, spec=0.2),
    "logo": make_mat("LogoRed", (78, 14, 22), 0.6),
    "logo_edge": make_mat("LogoEdge", (190, 150, 150), 0.6),
    "pants": make_mat("Pants", (42, 42, 45), 0.7, spec=0.3),
    "shoe": make_mat("Shoe", (232, 232, 230), 0.5),
    "sole": make_mat("ShoeSole", (200, 200, 198), 0.6),
    "chain": make_mat("GoldChain", (222, 190, 120), 0.25, metal=1.0),
}

# --------------------------------------------------------------------------- mesh helpers
PARTS = []  # (object, allowed bone list)


def bm_to_obj(bm, name, mat, bones, smooth=True):
    me = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    COLL.objects.link(ob)
    me.materials.append(MAT[mat])
    if smooth:
        me.shade_smooth()
    PARTS.append((ob, bones))
    return ob


def catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def resample(rings, sub):
    R = [np.array(r, dtype=float) for r in rings]
    if len(R) < 3 or sub <= 1:
        return R
    out = []
    for i in range(len(R) - 1):
        p0 = R[i - 1] if i > 0 else R[i]
        p3 = R[i + 2] if i + 2 < len(R) else R[i + 1]
        for k in range(sub):
            out.append(catmull(p0, R[i], R[i + 1], p3, k / sub))
    out.append(R[-1])
    return out


def loft(rings, axis="z", segs=20, p=2.0, cap0=True, cap1=True, sub=3, deform=None):
    """rings: (cx, cy, cz, a, b). For axis z: a->x radius, b->y radius;
    axis x: a->y, b->z; axis y: a->x, b->z. p = superellipse exponent."""
    rings = resample(rings, sub)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    grid = []
    n = len(rings)
    for i, (cx, cy, cz, a, b) in enumerate(rings):
        row = []
        for j in range(segs):
            th = 2 * math.pi * j / segs
            c, s = math.cos(th), math.sin(th)
            ce = math.copysign(abs(c) ** (2 / p), c)
            se = math.copysign(abs(s) ** (2 / p), s)
            if axis == "z":
                co = Vector((cx + a * ce, cy + b * se, cz))
            elif axis == "x":
                co = Vector((cx, cy + a * ce, cz + b * se))
            else:
                co = Vector((cx + a * ce, cy, cz + b * se))
            if deform:
                co = deform(co, i / (n - 1), th)
            row.append(bm.verts.new(co))
        grid.append(row)
    for i in range(n - 1):
        for j in range(segs):
            j2 = (j + 1) % segs
            f = bm.faces.new((grid[i][j], grid[i][j2], grid[i + 1][j2], grid[i + 1][j]))
            for l, (uu, vv) in zip(f.loops, ((j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1))):
                l[uv].uv = (uu / segs, vv / (n - 1))
    for idx, flag in ((0, cap0), (n - 1, cap1)):
        if not flag:
            continue
        row = grid[idx]
        ctr = bm.verts.new(sum((v.co for v in row), Vector()) / segs)
        for j in range(segs):
            bm.faces.new((row[j], row[(j + 1) % segs], ctr))
    return bm


def ellipsoid(center, radii, u=24, v=16, deform=None):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0, calc_uvs=True)
    for vt in bm.verts:
        co = vt.co.copy()
        if deform:
            co = deform(co)
        vt.co = Vector((center[0] + co.x * radii[0], center[1] + co.y * radii[1], center[2] + co.z * radii[2]))
    return bm


def torus(center, rx, ry, r, segs=48, rsegs=8, tilt_x=0.0):
    bm = bmesh.new()
    grid = []
    rot = Matrix.Rotation(tilt_x, 3, "X")
    for i in range(segs):
        th = 2 * math.pi * i / segs
        c = Vector((rx * math.cos(th), ry * math.sin(th), 0))
        out = Vector((math.cos(th), math.sin(th), 0))
        row = []
        for j in range(rsegs):
            ph = 2 * math.pi * j / rsegs
            co = c + out * (r * math.cos(ph)) + Vector((0, 0, r * math.sin(ph)))
            row.append(bm.verts.new(rot @ co + Vector(center)))
        grid.append(row)
    for i in range(segs):
        for j in range(rsegs):
            bm.faces.new((grid[i][j], grid[(i + 1) % segs][j], grid[(i + 1) % segs][(j + 1) % rsegs], grid[i][(j + 1) % rsegs]))
    return bm


def box(center, size, rot_y=0.0, bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.verts[:] + bm.edges[:], offset=bevel, segments=2, affect="EDGES")
    m = Matrix.Translation(center) @ Matrix.Rotation(rot_y, 4, "Y") @ Matrix.Diagonal((*size, 1))
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    return bm


def mirror_x(fn):
    """call fn(side) for side in (+1 left, -1 right); side name prefix."""
    for s, nm in ((1, "Left"), (-1, "Right")):
        fn(s, nm)


def bvh_of(bm):
    return BVHTree.FromBMesh(bm)


# =========================================================================== HEAD
HC = Vector((0, 0.005, 1.635))
HR = (0.080, 0.097, 0.112)


def head_deform(co):
    x, y, z = co
    if z < 0:  # jaw tapering to chin
        x *= 1 - 0.30 * z * z
        if y < 0:
            y *= 1 + 0.06 * (-z)
    else:
        x *= 1 + 0.03 * z
    if y > 0:
        y *= 1.07  # occiput
    if y < 0 and abs(x) > 0.55:  # flatten cheeks / face plane
        y *= 0.94
    if -0.15 < z < 0.25 and y < 0:  # cheekbones
        x *= 1.03
    return Vector((x, y, z))


head_bm = ellipsoid(HC, HR, 40, 28, head_deform)
head_bvh = bvh_of(head_bm)
bm_to_obj(head_bm.copy(), "Head", "skin", ["Head"])


def on_face(x, z, out=0.0):
    hit = head_bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)))
    loc, nrm = hit[0], hit[1]
    return loc + nrm * out, nrm


# neck
bm_to_obj(loft([(0, 0.012, 1.38, 0.058, 0.060), (0, 0.012, 1.46, 0.054, 0.056), (0, 0.010, 1.56, 0.052, 0.054)],
               "z", 20), "Neck", "skin", ["Neck", "Head", "Spine2"])

# ears
for s in (1, -1):
    bm_to_obj(ellipsoid((s * 0.081, 0.012, 1.632), (0.013, 0.026, 0.033), 12, 10), "Ear", "skin", ["Head"])

# nose
loc, n = on_face(0, 1.622)
bm_to_obj(ellipsoid(loc + Vector((0, -0.006, 0)), (0.009, 0.012, 0.028), 14, 10), "NoseBridge", "skin", ["Head"])
loc, n = on_face(0, 1.598)
bm_to_obj(ellipsoid(loc + Vector((0, -0.010, 0.002)), (0.017, 0.012, 0.010), 14, 10), "NoseTip", "skin", ["Head"])

# eyes, brows
for s in (1, -1):
    loc, n = on_face(s * 0.031, 1.652)
    bm_to_obj(ellipsoid(loc + Vector((0, 0.004, 0)), (0.013, 0.006, 0.0065), 14, 10), "EyeWhite", "eye_white", ["Head"])
    bm_to_obj(ellipsoid(loc + Vector((0, -0.0005, 0)), (0.0062, 0.003, 0.0062), 12, 8), "Iris", "iris", ["Head"])
    # upper lid / brow ridge (skin) to give the frowning look
    loc2, _ = on_face(s * 0.031, 1.663)
    bm_to_obj(ellipsoid(loc2 + Vector((0, -0.001, 0)), (0.017, 0.006, 0.0045), 12, 8), "Lid", "skin", ["Head"])
    loc3, _ = on_face(s * 0.035, 1.674)
    b = box(loc3 + Vector((0, -0.002, 0)), (0.034, 0.008, 0.0065), rot_y=-s * 0.30, bevel=0.002)
    bm_to_obj(b, "Brow", "brow", ["Head"], smooth=False)

# mouth: lips + thin mustache
loc, _ = on_face(0, 1.574)
bm_to_obj(ellipsoid(loc + Vector((0, 0.001, 0)), (0.022, 0.006, 0.006), 14, 8), "Lips", "lips", ["Head"])
must = []
for k in range(7):
    x = -0.026 + k * 0.026 / 3
    z = 1.587 - 18 * x * x
    p, _ = on_face(x, z, 0.002)
    must.append((p.x, p.y, p.z, 0.0035, 0.0028))
bm_to_obj(loft(must, "x", 8, sub=2), "Mustache", "brow", ["Head"])

# =========================================================================== HAIR
# short fade on the sides / back
fade = ellipsoid(HC, (HR[0] * 1.035, HR[1] * 1.035, HR[2] * 1.03), 72, 48, head_deform)
kill = []
for f in fade.faces:
    c = f.calc_center_median() - HC
    zl, yl = c.z / HR[2], c.y / HR[1]
    keep = (zl > 0.15 and yl > -0.62) or (zl > -0.25 and yl > 0.05) or (zl > 0.62)
    if not keep:
        kill.append(f)
bmesh.ops.delete(fade, geom=kill, context="FACES")
bm_to_obj(fade, "HairFade", "fade", ["Head"])

# puff on top
PC = Vector((0, -0.004, 1.728))
PR = (0.088, 0.106, 0.050)
bm_to_obj(ellipsoid(PC, PR, 24, 14), "HairPuff", "hair", ["Head"])

# twisted spikes (high-top twists)
spk = bmesh.new()
for i in range(200):
    # sample the upper part of the puff
    th = random.uniform(0, 2 * math.pi)
    ph = math.acos(random.uniform(0.05, 1.0))  # polar angle from +z
    d = Vector((math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)))
    base = PC + Vector((d.x * PR[0], d.y * PR[1], d.z * PR[2])) * 0.85
    dirv = (Vector((d.x / PR[0], d.y / PR[1], d.z / PR[2])).normalized() + Vector((0, 0, 0.55))
            + Vector((random.uniform(-.3, .3), random.uniform(-.3, .3), random.uniform(-.1, .2)))).normalized()
    L = random.uniform(0.035, 0.065) * (1.2 if d.y < -0.3 else 1.0)
    r = random.uniform(0.015, 0.023)
    tmp = bmesh.new()
    bmesh.ops.create_cone(tmp, cap_ends=True, segments=7, radius1=r, radius2=0.003, depth=L)
    # twist the cone a bit for a "dread twist" look
    for v in tmp.verts:
        t = (v.co.z + L / 2) / L
        a = t * 1.6
        v.co.x, v.co.y = v.co.x * math.cos(a) - v.co.y * math.sin(a), v.co.x * math.sin(a) + v.co.y * math.cos(a)
        v.co.x += 0.006 * t * t  # slight curl
    rot = Vector((0, 0, 1)).rotation_difference(dirv).to_matrix().to_4x4()
    bmesh.ops.transform(tmp, matrix=Matrix.Translation(base + dirv * (L / 2)) @ rot, verts=tmp.verts)
    me = bpy.data.meshes.new("tmp")
    tmp.to_mesh(me)
    tmp.free()
    spk.from_mesh(me)
    bpy.data.meshes.remove(me)
bm_to_obj(spk, "HairTwists", "hair", ["Head"], smooth=False)

# =========================================================================== SHIRT (oversized tee)
def shirt_wrinkle(co, t, th):
    # soft vertical folds + slight drape
    k = 0.004 * math.sin(th * 7 + 1.3) * (1 - t)
    d = Vector((co.x, co.y * 1.6, 0))
    if d.length > 0:
        co += d.normalized() * k
    return co


torso_rings = [
    (0, 0.005, 0.785, 0.275, 0.158),
    (0, 0.005, 0.90, 0.277, 0.160),
    (0, 0.005, 1.05, 0.272, 0.158),
    (0, 0.008, 1.20, 0.268, 0.152),
    (0, 0.010, 1.31, 0.262, 0.140),
    (0, 0.012, 1.395, 0.235, 0.122),
    (0, 0.012, 1.44, 0.165, 0.100),
    (0, 0.012, 1.462, 0.072, 0.072),
]
shirt_bm = loft(torso_rings, "z", 32, p=3.0, cap0=False, cap1=False, deform=shirt_wrinkle)
shirt_bvh = bvh_of(shirt_bm)
TORSO_BONES = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "LeftShoulder", "RightShoulder", "LeftArm", "RightArm"]
bm_to_obj(shirt_bm, "ShirtBody", "shirt", TORSO_BONES)
# hem band + stitch lines
bm_to_obj(loft([(0, 0.005, 0.783, 0.279, 0.162), (0, 0.005, 0.80, 0.279, 0.162)], "z", 32, p=3.0, cap0=False, cap1=False),
          "ShirtHem", "shirt", TORSO_BONES)
for z in (0.792, 0.808):
    bm_to_obj(loft([(0, 0.005, z - 0.0018, 0.2805, 0.1635), (0, 0.005, z + 0.0018, 0.2805, 0.1635)], "z", 32, p=3.0,
                   cap0=False, cap1=False, sub=1), "HemStitch", "trim", TORSO_BONES)
# collar (ribbed)
bm_to_obj(torus((0, 0.012, 1.462), 0.074, 0.074, 0.008, 40, 8), "Collar", "shirt", ["Spine2", "Neck"])

# sleeves (wide, short, dropped)
for s, nm in ((1, "Left"), (-1, "Right")):
    sl = [(s * x, 0.008, z, a, b) for x, z, a, b in (
        (0.16, 1.365, 0.118, 0.105), (0.24, 1.355, 0.112, 0.102), (0.33, 1.345, 0.104, 0.096),
        (0.40, 1.338, 0.098, 0.092), (0.445, 1.335, 0.096, 0.090))]
    bm_to_obj(loft(sl, "x", 24, cap0=False, cap1=False), nm + "Sleeve", "shirt",
              [nm + "Shoulder", nm + "Arm", "Spine2"])
    for x in (0.425, 0.437):
        band = [(s * (x - 0.0018), 0.008, 1.335, 0.0985, 0.0925), (s * (x + 0.0018), 0.008, 1.335, 0.0985, 0.0925)]
        bm_to_obj(loft(band, "x", 24, cap0=False, cap1=False, sub=1), nm + "CuffStitch", "trim", [nm + "Arm"])

# gold chain (thin, visible at the collar)
bm_to_obj(torus((0, 0.000, 1.452), 0.085, 0.090, 0.0028, 64, 6, tilt_x=0.30), "Chain", "chain", ["Spine2", "Neck"])

# logo "Solatnill"
def make_logo():
    cu = bpy.data.curves.new("LogoText", "FONT")
    cu.body = "Solatnill"
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.size = 0.1
    if args.font and os.path.exists(args.font):
        cu.font = bpy.data.fonts.load(args.font)
    layers = []
    for nm, mat, off, ext, dy in (("LogoEdge", "logo_edge", 0.0012, 0.0008, 0.0006), ("Logo", "logo", 0.0, 0.0010, -0.0004)):
        c = cu.copy()
        c.offset = off
        c.extrude = ext
        o = bpy.data.objects.new("tmp_txt", c)
        COLL.objects.link(o)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
        bpy.data.objects.remove(o)
        layers.append((nm, mat, me, dy))
    # common scale from the base text width
    xs = [v.co.x for v in layers[1][2].vertices]
    scale = 0.34 / (max(xs) - min(xs))
    for nm, mat, me, dy in layers:
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
        for v in bm.verts:
            x, y, z = v.co.x * scale, v.co.y * scale, v.co.z
            wz = 1.215 + y
            hit = shirt_bvh.ray_cast(Vector((x, -1.0, wz)), Vector((0, 1, 0)))
            sy = hit[0].y if hit[0] else -0.16
            v.co = Vector((x, sy - 0.0012 + dy - z, wz))
        bm_to_obj(bm, nm, mat, TORSO_BONES, smooth=False)


make_logo()

# =========================================================================== ARMS / HANDS
for s, nm in ((1, "Left"), (-1, "Right")):
    arm = [(s * x, 0.010, z, a, b) for x, z, a, b in (
        (0.15, 1.385, 0.056, 0.060), (0.24, 1.384, 0.052, 0.055), (0.34, 1.384, 0.046, 0.047),
        (0.44, 1.384, 0.041, 0.040), (0.475, 1.384, 0.038, 0.037), (0.53, 1.385, 0.042, 0.039),
        (0.60, 1.386, 0.037, 0.031), (0.66, 1.387, 0.030, 0.024), (0.695, 1.387, 0.028, 0.020))]
    bm_to_obj(loft(arm, "x", 18), nm + "Arm", "skin", [nm + "Shoulder", nm + "Arm", nm + "ForeArm", nm + "Hand"])
    hb = [nm + "Hand", nm + "ForeArm"]
    # palm (palm facing down)
    palm = [(s * x, 0.010, 1.386, a, b) for x, a, b in (
        (0.685, 0.030, 0.020), (0.72, 0.042, 0.020), (0.76, 0.046, 0.018), (0.795, 0.044, 0.015))]
    bm_to_obj(loft(palm, "x", 16), nm + "Palm", "skin", hb)
    # fingers (index at the front = -Y)
    for fy, ln in ((-0.030, 0.082), (-0.010, 0.092), (0.010, 0.088), (0.029, 0.070)):
        fing = []
        spread = fy * 0.35
        for k in range(5):
            t = k / 4
            x = 0.785 + t * ln
            fing.append((s * x, 0.010 + fy + spread * t, 1.386 - 0.012 * t * t, 0.0095 - 0.002 * t, 0.0085 - 0.002 * t))
        bm_to_obj(loft(fing, "x", 10, sub=2), nm + "Finger", "skin", hb)
    thumb = []
    for k in range(5):
        t = k / 4
        thumb.append((s * (0.715 + 0.050 * t), 0.010 - 0.035 - 0.040 * t, 1.383 - 0.006 * t, 0.012 - 0.003 * t, 0.011 - 0.002 * t))
    # thumb runs diagonally, approximate with an x-axis loft
    bm_to_obj(loft(thumb, "x", 10, sub=2), nm + "Thumb", "skin", hb)

# =========================================================================== PANTS (baggy)
def pants_folds(co, t, th):
    # stacked folds near the hem and soft folds at the knee
    k = 0.0
    if t < 0.25:
        k += 0.010 * math.sin(th * 3 + t * 55) * (0.25 - t) * 4
    k += 0.004 * math.sin(th * 5 + t * 18)
    co.x += math.cos(th) * k
    co.y += math.sin(th) * k
    return co


for s, nm in ((1, "Left"), (-1, "Right")):
    leg = [(s * x, y, z, a, b) for x, y, z, a, b in (
        (0.112, 0.010, 0.040, 0.118, 0.135), (0.112, 0.008, 0.10, 0.112, 0.126), (0.110, 0.004, 0.25, 0.108, 0.118),
        (0.108, 0.000, 0.45, 0.112, 0.122), (0.106, 0.004, 0.65, 0.118, 0.130), (0.100, 0.006, 0.82, 0.122, 0.135))]
    bm_to_obj(loft(leg, "z", 24, cap0=True, cap1=True, deform=pants_folds), nm + "PantLeg", "pants",
              [nm + "UpLeg", nm + "Leg", "Hips"])
bm_to_obj(loft([(0, 0.006, 0.66, 0.205, 0.120), (0, 0.006, 0.78, 0.225, 0.135), (0, 0.006, 0.93, 0.225, 0.135)],
               "z", 28, p=2.5), "PantsPelvis", "pants", ["Hips", "LeftUpLeg", "RightUpLeg", "Spine"])

# =========================================================================== SHOES (white low-top sneakers)
for s, nm in ((1, "Left"), (-1, "Right")):
    x0 = s * 0.112
    up = [(x0, y, z, a, b) for y, z, a, b in (
        (-0.175, 0.038, 0.030, 0.020), (-0.160, 0.042, 0.048, 0.030), (-0.110, 0.050, 0.056, 0.040),
        (-0.040, 0.062, 0.057, 0.058), (0.040, 0.066, 0.054, 0.066), (0.090, 0.064, 0.050, 0.060),
        (0.112, 0.058, 0.036, 0.045))]
    bm_to_obj(loft(up, "y", 20), nm + "Shoe", "shoe", [nm + "Foot", nm + "ToeBase"])
    sole = [(x0, y, 0.016, a, 0.016) for y, a in (
        (-0.188, 0.030), (-0.175, 0.055), (-0.10, 0.064), (0.0, 0.060), (0.08, 0.058), (0.122, 0.040))]
    bm_to_obj(loft(sole, "y", 20, p=3.5), nm + "Sole", "sole", [nm + "Foot", nm + "ToeBase"])

# =========================================================================== ARMATURE
JOINTS = {
    # name: (head, tail, parent)
    "Hips": ((0, 0.01, 0.94), (0, 0.01, 1.04), None),
    "Spine": ((0, 0.01, 1.04), (0, 0.01, 1.16), "Hips"),
    "Spine1": ((0, 0.01, 1.16), (0, 0.01, 1.28), "Spine"),
    "Spine2": ((0, 0.01, 1.28), (0, 0.012, 1.43), "Spine1"),
    "Neck": ((0, 0.012, 1.43), (0, 0.010, 1.54), "Spine2"),
    "Head": ((0, 0.010, 1.54), (0, 0.010, 1.80), "Neck"),
}
for s, nm in ((1, "Left"), (-1, "Right")):
    JOINTS.update({
        nm + "Shoulder": ((s * 0.03, 0.01, 1.40), (s * 0.165, 0.01, 1.388), "Spine2"),
        nm + "Arm": ((s * 0.165, 0.01, 1.388), (s * 0.47, 0.01, 1.385), nm + "Shoulder"),
        nm + "ForeArm": ((s * 0.47, 0.01, 1.385), (s * 0.69, 0.01, 1.387), nm + "Arm"),
        nm + "Hand": ((s * 0.69, 0.01, 1.387), (s * 0.80, 0.01, 1.386), nm + "ForeArm"),
        nm + "UpLeg": ((s * 0.105, 0.01, 0.90), (s * 0.108, 0.0, 0.50), "Hips"),
        nm + "Leg": ((s * 0.108, 0.0, 0.50), (s * 0.112, 0.03, 0.085), nm + "UpLeg"),
        nm + "Foot": ((s * 0.112, 0.03, 0.085), (s * 0.112, -0.10, 0.030), nm + "Leg"),
        nm + "ToeBase": ((s * 0.112, -0.10, 0.030), (s * 0.112, -0.175, 0.030), nm + "Foot"),
    })

arm_data = bpy.data.armatures.new("SolatnillRig")
rig = bpy.data.objects.new("SolatnillRig", arm_data)
COLL.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for name, (h, t, par) in JOINTS.items():
    b = arm_data.edit_bones.new(name)
    b.head, b.tail = Vector(h), Vector(t)
    b.roll = 0.0
    eb[name] = b
for name, (h, t, par) in JOINTS.items():
    if par:
        eb[name].parent = eb[par]
        eb[name].use_connect = (Vector(h) - eb[par].tail).length < 1e-4
bpy.ops.object.mode_set(mode="OBJECT")

SEG = {n: (np.array(h), np.array(t)) for n, (h, t, _) in JOINTS.items()}


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


for ob, bones in PARTS:
    me = ob.data
    P = np.array([v.co[:] for v in me.vertices])
    D = np.stack([seg_dist(P, *SEG[b]) for b in bones], axis=1) + 1e-4
    W = 1.0 / D ** 5
    # keep the 3 strongest influences
    if W.shape[1] > 3:
        thr = np.sort(W, axis=1)[:, -3][:, None]
        W = np.where(W >= thr, W, 0)
    W /= W.sum(axis=1, keepdims=True)
    groups = {b: ob.vertex_groups.new(name=b) for b in bones}
    for j, b in enumerate(bones):
        g = groups[b]
        for i in np.nonzero(W[:, j] > 1e-3)[0]:
            g.add([int(i)], float(W[i, j]), "REPLACE")

# --------------------------------------------------------------------------- join into one skinned mesh
bpy.ops.object.select_all(action="DESELECT")
for ob, _ in PARTS:
    ob.select_set(True)
body = PARTS[0][0]
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
body.name = body.data.name = "Solatnill"
# merge duplicate material slots created by joining
mats = {}
for i, slot in enumerate(body.material_slots):
    mats.setdefault(slot.material.name, i)
remap = [mats[s.material.name] for s in body.material_slots]
for poly in body.data.polygons:
    poly.material_index = remap[poly.material_index]
bpy.ops.object.material_slot_remove_unused()

body.parent = rig
mod = body.modifiers.new("Armature", "ARMATURE")
mod.object = rig

tris = sum(len(p.vertices) - 2 for p in body.data.polygons)
print(f"[solatnill] verts={len(body.data.vertices)} tris={tris} materials={len(body.material_slots)}")

# --------------------------------------------------------------------------- export
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "Solatnill.blend"))
bpy.ops.export_scene.fbx(
    filepath=os.path.join(OUT, "Solatnill.fbx"),
    use_selection=False,
    object_types={"ARMATURE", "MESH"},
    apply_unit_scale=True,
    apply_scale_options="FBX_SCALE_UNITS",
    add_leaf_bones=False,
    bake_anim=False,
    mesh_smooth_type="FACE",
    path_mode="AUTO",
)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "Solatnill.glb"), export_format="GLB")

# --------------------------------------------------------------------------- previews
if args.render:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("W")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.052, 0.056, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    for loc, energy, size in (((-2.5, -3.5, 3.0), 900, 3.0), ((3.0, -2.0, 2.0), 350, 3.0), ((0, 4.0, 3.0), 500, 3.0)):
        ld = bpy.data.lights.new("L", "AREA")
        ld.energy, ld.size = energy, size
        lo = bpy.data.objects.new("L", ld)
        lo.location = loc
        lo.rotation_euler = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        COLL.objects.link(lo)
    cd = bpy.data.cameras.new("Cam")
    cd.type = "ORTHO"
    cd.ortho_scale = 2.0
    cam = bpy.data.objects.new("Cam", cd)
    COLL.objects.link(cam)
    scene.camera = cam
    os.makedirs(os.path.join(OUT, "previews"), exist_ok=True)
    views = {
        "front": ((0, -6, 0.92), (math.pi / 2, 0, 0)),
        "left": ((6, 0, 0.92), (math.pi / 2, 0, math.pi / 2)),
        "back": ((0, 6, 0.92), (math.pi / 2, 0, math.pi)),
        "right": ((-6, 0, 0.92), (math.pi / 2, 0, -math.pi / 2)),
    }
    for nm, (loc, rot) in views.items():
        cam.location, cam.rotation_euler = loc, rot
        scene.render.filepath = os.path.join(OUT, "previews", nm + ".png")
        bpy.ops.render.render(write_still=True)
    # close-up of the face
    cd.ortho_scale = 0.45
    cam.location, cam.rotation_euler = (0, -6, 1.66), (math.pi / 2, 0, 0)
    scene.render.filepath = os.path.join(OUT, "previews", "face.png")
    bpy.ops.render.render(write_still=True)
print("[solatnill] done ->", OUT)
