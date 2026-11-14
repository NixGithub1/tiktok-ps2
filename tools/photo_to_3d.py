"""Turn a front-view character photo on a plain background into a textured 3D mesh (GLB/OBJ).

Method: the character is cut out from the uniform background, then "inflated"
into a volume (depth derived from the distance to the silhouette edge). The
front of the model is textured with the photo itself, unchanged; the back
reuses the photo with the face replaced by hair and the chest print removed.

Usage: python3 tools/photo_to_3d.py <image> <out_dir> [--height 1.80] [--step 2]
"""
import argparse
import io
import os

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage as ndi


def segment(img):
    h, w, _ = img.shape
    border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3),
                             img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    smooth = ndi.median_filter(img, size=(5, 5, 1))
    diff = np.abs(smooth - bg).sum(2)
    sat = smooth.max(2) - smooth.min(2)
    # background = near-uniform pixels connected to the image border
    near_bg = (diff < 22) & (sat < 12)
    lab, _ = ndi.label(near_bg)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    background = np.isin(lab, edge_labels[edge_labels > 0])
    mask = ~background
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = lab == (np.argmax(sizes) + 1)
    return mask, bg


def back_texture(img, mask):
    """Same colors as the photo, but seen from behind: no face, no chest print."""
    back = img.copy()
    ys, xs = np.nonzero(mask)
    top = ys.min()
    rows = mask.sum(1)
    # head = narrow top part until the silhouette widens (shoulders)
    head_bottom = top
    for y in range(top, top + 400):
        if rows[y] > 3.0 * rows[top + 40]:
            break
        head_bottom = y
    neck = int(top + 0.80 * (head_bottom - top))
    r, g, b = [img[..., i] for i in range(3)]
    lum = img.mean(2)
    head = np.zeros_like(mask)
    head[top:neck] = mask[top:neck]
    hair = head & (lum < 60)
    hair_col = np.median(img[hair], axis=0) if hair.any() else np.array([20, 20, 20])
    back[head] = hair_col
    # chest print: reddish dark pixels inside the shirt band
    shirt_band = np.zeros_like(mask)
    shirt_band[head_bottom:head_bottom + 330] = mask[head_bottom:head_bottom + 330]
    light = shirt_band & (lum > 170)
    shirt_col = np.median(img[light], axis=0)
    cx = int(np.median(xs))
    logo = shirt_band & (r > g + 12) & (r > b + 8) & (lum < 190)
    logo[:, :cx - 160] = False
    logo[:, cx + 160:] = False
    logo = ndi.binary_closing(logo, iterations=3)
    logo = ndi.binary_dilation(logo, iterations=6) & mask
    back[logo] = shirt_col
    # necklace and other small front details get softened by a slight blur
    blurred = ndi.uniform_filter(back, size=(5, 5, 1))
    back[mask] = blurred[mask]
    return back


def build_mesh(img, mask, step, height_m):
    h, w, _ = img.shape
    dt = ndi.distance_transform_edt(mask)
    depth = np.sqrt(dt) * 5.5
    depth = ndi.gaussian_filter(depth, 3) * mask
    depth = np.maximum(depth, 0.6) * mask

    gm = mask[::step, ::step]
    gd = depth[::step, ::step]
    gh, gw = gm.shape
    idx = -np.ones((gh, gw), int)
    pts = np.argwhere(gm)
    idx[pts[:, 0], pts[:, 1]] = np.arange(len(pts))
    n = len(pts)

    ys, xs = np.nonzero(mask)
    px_h = ys.max() - ys.min()
    s = height_m / px_h
    cx = (xs.min() + xs.max()) / 2
    row = pts[:, 0] * step
    col = pts[:, 1] * step
    z = gd[pts[:, 0], pts[:, 1]]
    X = (col - cx) * s
    Y = (ys.max() - row) * s
    front = np.c_[X, Y, z * s]
    back = np.c_[X, Y, -z * s]
    verts = np.vstack([front, back])
    # texture atlas: front photo on the left half, back on the right half
    u = col / (w - 1)
    v = 1 - row / (h - 1)
    uv = np.vstack([np.c_[u * 0.5, v], np.c_[0.5 + (1 - u) * 0.5, v]])

    a = idx[:-1, :-1]; b = idx[:-1, 1:]; c = idx[1:, 1:]; d = idx[1:, :-1]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d = a[ok], b[ok], c[ok], d[ok]
    faces_f = np.vstack([np.c_[a, d, c], np.c_[a, c, b]])
    faces_b = np.vstack([np.c_[a, c, d], np.c_[a, b, c]]) + n
    faces = np.vstack([faces_f, faces_b])

    # stitch the silhouette edge (edges used by exactly one front triangle)
    edges = np.vstack([faces_f[:, [0, 1]], faces_f[:, [1, 2]], faces_f[:, [2, 0]]])
    key = np.sort(edges, 1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    border = edges[cnt[inv.ravel()] == 1]
    p, q = border[:, 0], border[:, 1]
    sides = np.vstack([np.c_[q, p, p + n], np.c_[q, p + n, q + n]])
    faces = np.vstack([faces, sides])

    # keep every vertex (even unused ones) so uv rows stay aligned with vertices
    mesh = trimesh.Trimesh(verts, faces, process=False)
    return mesh, uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--height", type=float, default=1.80)
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    img = np.array(Image.open(args.image).convert("RGB"))
    mask, _ = segment(img.astype(int))
    back = back_texture(img.astype(float), mask).clip(0, 255).astype(np.uint8)
    atlas = np.concatenate([img, back[:, ::-1]], axis=1)
    atlas_img = Image.fromarray(atlas)
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out_dir, "mask.png"))
    atlas_img.save(os.path.join(args.out_dir, "texture.jpg"), quality=92)

    mesh, uv = build_mesh(img, mask, args.step, args.height)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas_img, metallicFactor=0.0,
                                             roughnessFactor=0.9, doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.export(os.path.join(args.out_dir, "personnage.glb"))
    simple = trimesh.visual.material.SimpleMaterial(image=atlas_img)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=simple)
    mesh.export(os.path.join(args.out_dir, "personnage.obj"))
    print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"size={mesh.extents.round(3)} m watertight={mesh.is_watertight}")


if __name__ == "__main__":
    main()
