"""Druckdaten: alle vier Schablonen als STL für ein Druckbett, Vierfarbbild als PNG."""
import struct
import cv2
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry.polygon import orient
from stencil_geom import PLATE_W, PLATE_H

BED = 250.0        # mm, quadratisches Druckbett
HEIGHT = 0.5       # mm
EDGE, GAP = 2.0, 4.0
PAPER = (255, 255, 255)   # Weiß: unbesprühtes Papier unter allen geschlossenen Platten


def layout(bed=BED):
    """Maßstab und Ursprung (links unten) der vier Platten im 2 × 2-Raster, S1 oben links."""
    cell = (bed - 2 * EDGE - GAP) / 2
    s = min(cell / PLATE_W, cell / PLATE_H)
    w, h = PLATE_W * s, PLATE_H * s
    x0, y0 = (bed - 2 * w - GAP) / 2, (bed - 2 * h - GAP) / 2
    return s, [(x0 + (i % 2) * (w + GAP), y0 + (1 - i // 2) * (h + GAP)) for i in range(4)]


def prism(poly, h):
    """Polygon (CCW außen) -> Dreiecke eines geschlossenen Prismas, Array (n, 3, 3)."""
    tris = []
    for t in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)):
        c = np.array(t.exterior.coords[:3])
        (ax, ay), (bx, by) = c[1] - c[0], c[2] - c[0]
        if ax * by - ay * bx < 0:                            # im Uhrzeigersinn: umdrehen
            c = c[::-1]
        tris.append(np.c_[c, np.full(3, h)])                 # oben, Normale +z
        tris.append(np.c_[c[::-1], np.zeros(3)])             # unten, Normale -z
    for r in [poly.exterior, *poly.interiors]:               # außen CCW, Löcher CW: rechts liegt außen
        c = np.array(r.coords)
        a, b = c[:-1], c[1:]
        z0, z1 = np.zeros((len(a), 1)), np.full((len(a), 1), h)
        A0, B0, A1, B1 = np.hstack([a, z0]), np.hstack([b, z0]), np.hstack([a, z1]), np.hstack([b, z1])
        tris += list(np.stack([A0, B0, B1], 1)) + list(np.stack([A0, B1, A1], 1))
    return np.array(tris)


def stl(res, bed=BED, height=HEIGHT):
    """Binäres STL mit allen Platten, in mm. Liefert (bytes, Maßstab, Plattenmaß)."""
    s, origins = layout(bed)
    parts = []
    for r, (ox, oy) in zip(res, origins):
        g = affinity.scale(r["solid"], s, -s, origin=(0, 0))            # Draufsicht: y nach oben
        g = affinity.translate(g, ox, oy + PLATE_H * s)
        g = g.simplify(.02)
        for p in getattr(g, "geoms", [g]):
            if p.area > 0:
                parts.append(prism(orient(p, 1.0), height))
    t = np.concatenate(parts).astype(np.float32)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, t
    head = f"Schablonen 4x {PLATE_W * s:.1f}x{PLATE_H * s:.1f}x{height} mm".encode().ljust(80, b" ")
    return head + struct.pack("<I", len(t)) + rec.tobytes(), s


def bild(res):
    """Vierfarbbild des Motivfelds aus den geschnittenen Öffnungen, PNG im Arbeitsraster (6 px/mm)."""
    img = np.empty(res[0]["mask"].shape + (3,), np.uint8)
    img[:] = PAPER[::-1]
    for r in res:
        img[r["mask"]] = r["rgb"][::-1]
    return cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])[1].tobytes()
