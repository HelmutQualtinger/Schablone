"""Rastermasken -> zusammenhängende Schablonen (Stege) -> Vektorgeometrie in mm."""
import heapq
import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from motiv import separate, PX_MM, MOTIF_W, MOTIF_H, PREFIX

PLATE_W, PLATE_H, MARGIN = 420, 510, 30       # mm
THICKNESS = 1.0
MIN_WEB = 2.0                                 # kleinster Steg / kleinste Materialbreite
REG_HOLES = [(15, 15), (405, 15), (15, 495), (380, 495)]   # asymmetrisch: verhindert Wenden
REG_D = 6.0
NOTCH_W, NOTCH_D, NOTCH_PITCH = 6, 5, 12
ROUTE_PX_MM = 2

LAYERS = [
    dict(id="S1", name="Sonnengelb", key="yellow", rgb=(245, 203, 38), ral="RAL 1018 (nahe)", file=f"{PREFIX}_S1_gelb"),
    dict(id="S2", name="Korallrosa", key="rose", rgb=(235, 110, 122), ral="RAL 3014 (nahe)", file=f"{PREFIX}_S2_rosa"),
    dict(id="S3", name="Petrol", key="petrol", rgb=(12, 86, 100), ral="RAL 5020 (nahe)", file=f"{PREFIX}_S3_petrol"),
    dict(id="S4", name="Tiefschwarz", key="black", rgb=(22, 24, 28), ral="RAL 9005", file=f"{PREFIX}_S4_schwarz"),
]


def disk(mm):
    r = max(1, round(mm * PX_MM))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def morph(m, op, mm):
    return cv2.morphologyEx(m.astype(np.uint8), op, disk(mm)) > 0


def drop_small(m, mm2):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    keep = st[:, cv2.CC_STAT_AREA] >= mm2 * PX_MM ** 2
    keep[0] = False
    return keep[lab]


def clean(o):
    o = morph(o, cv2.MORPH_OPEN, MIN_WEB / 2)      # Öffnungen mind. 2 mm breit
    o = morph(o, cv2.MORPH_CLOSE, MIN_WEB / 2)     # Material mind. 2 mm breit
    o = drop_small(o, 14)
    return ~drop_small(~o, 40)


def route_bridges(opening, hidden):
    """Verbindet alle Materialinseln mit dem Rahmen.

    Inseln werden am erodierten Material bestimmt, damit auch Engstellen unter
    der Mindestbreite als getrennt gelten. Wege laufen bevorzugt dort, wo die
    nächste Sprühebene die Steglücke wieder abdeckt.
    """
    h, w = opening.shape
    pad = MARGIN * PX_MM
    solid = np.ones((h + 2 * pad, w + 2 * pad), np.uint8)
    solid[pad:-pad, pad:-pad] = ~opening
    hid = np.zeros_like(solid)
    hid[pad:-pad, pad:-pad] = hidden
    core = cv2.erode(solid, disk(MIN_WEB * .6))

    f = ROUTE_PX_MM / PX_MM
    small = lambda a: cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) > .5
    core_s, hid_s = small(core), small(cv2.erode(hid, disk(2)))
    H, W = core_s.shape
    n, lab = cv2.connectedComponents(core_s.astype(np.uint8), connectivity=4)
    main = lab[0, 0]
    cost = np.where(hid_s, 1.0, 9.0).ravel()
    lab_f = lab.ravel()
    members = {i: np.flatnonzero(lab_f == i) for i in range(1, n)}

    dist = np.full(H * W, np.inf)
    prev = np.full(H * W, -1, np.int64)
    heap = []
    done = {main}
    for p in members[main]:
        dist[p] = 0
        heap.append((0.0, int(p)))
    heapq.heapify(heap)
    paths = []
    while heap:
        d, p = heapq.heappop(heap)
        if d > dist[p]:
            continue
        l = lab_f[p]
        if l and l not in done:
            done.add(l)
            path, q = [], p
            while q >= 0 and dist[q] > 0:
                path.append(q)
                q = prev[q]
            area = len(members[l]) / ROUTE_PX_MM ** 2
            paths.append((path + [q], area))
            for m in members[l]:
                dist[m] = 0
                prev[m] = -1
                heapq.heappush(heap, (0.0, int(m)))
            d = 0
        y, x = p // W, p % W
        for dy, dx, s in ((0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (1, 1, 1.41), (1, -1, 1.41), (-1, 1, 1.41), (-1, -1, 1.41)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W:
                q = yy * W + xx
                c = 0 if lab_f[q] in done and lab_f[q] else cost[q] * s
                if d + c < dist[q]:
                    dist[q] = d + c
                    prev[q] = p
                    heapq.heappush(heap, (d + c, q))

    bridged = solid.copy()
    lengths = []
    for path, area in paths:
        pts = np.array([[(q % W), (q // W)] for q in path if q >= 0]) * PX_MM / ROUTE_PX_MM
        width = 3.0 if area < 400 else 4.5 if area < 2500 else 7.0
        cv2.polylines(bridged, [pts.astype(np.int32)], False, 1, round(width * PX_MM), cv2.LINE_8)
        lengths.append(len(pts) / ROUTE_PX_MM)
    out = ~(bridged[pad:-pad, pad:-pad] > 0)
    out = morph(out, cv2.MORPH_OPEN, MIN_WEB / 2)   # verkleinert nur Öffnungen, Stege bleiben
    out = drop_small(out, 14)
    return out, len(paths)


def chaikin(pts, it=2):
    for _ in range(it):
        a, b = pts, np.roll(pts, -1, 0)
        pts = np.empty((len(a) * 2, 2))
        pts[::2], pts[1::2] = .75 * a + .25 * b, .25 * a + .75 * b
    return pts


def vectorize(o):
    """Öffnungsmaske -> Liste shapely-Polygone in Plattenkoordinaten (mm, y nach unten)."""
    cs, hier = cv2.findContours(o.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for i, c in enumerate(cs):
        if hier[0][i][3] >= 0 or len(c) < 8:
            continue
        def ring(c):
            p = Polygon((c[:, 0] + .5) / PX_MM + MARGIN).simplify(.14).segmentize(2.0)   # lange Kanten: Chaikin kappt sonst Ecken
            return chaikin(np.array(p.exterior.coords[:-1])) if p.area > 0 else None
        ext = ring(c)
        holes = []
        j = hier[0][i][2]
        while j >= 0:
            if len(cs[j]) >= 8:
                r = ring(cs[j])
                if r is not None:
                    holes.append(r)
            j = hier[0][j][0]
        polys.append(Polygon(ext, holes).buffer(0))
    box_motif = box(MARGIN, MARGIN, MARGIN + MOTIF_W, MARGIN + MOTIF_H)
    return unary_union(polys).intersection(box_motif)


def fixtures(layer_no):
    plate = box(0, 0, PLATE_W, PLATE_H)
    holes = [Point(p).buffer(REG_D / 2, 32) for p in REG_HOLES]
    x0 = PLATE_W / 2 - (layer_no - 1) * NOTCH_PITCH / 2
    notches = [box(x0 + i * NOTCH_PITCH - NOTCH_W / 2, 0, x0 + i * NOTCH_PITCH + NOTCH_W / 2, NOTCH_D)
               for i in range(layer_no)]
    return plate, holes, notches


def build():
    m, _ = separate()
    fig_black = clean(m["black"])
    o3 = clean(m["petrol"] | fig_black)
    fig = ~m["petrol"] | m["yellow"] | m["rose"]
    o2 = clean(m["rose"] | (o3 & (m["yellow"] | m["rose"])) | (fig_black))
    o1 = clean(m["yellow"] | o2)
    raw = dict(S1=o1, S2=o2, S3=o3, S4=fig_black)
    next_layer = dict(S1=o2, S2=o3, S3=fig_black, S4=np.zeros_like(o1))

    res = []
    for no, L in enumerate(LAYERS, 1):
        o, n_bridges = route_bridges(raw[L["id"]], next_layer[L["id"]])
        opening = vectorize(o)
        plate, holes, notches = fixtures(no)
        solid = plate.difference(opening).difference(unary_union(holes + notches))
        res.append(dict(L, no=no, mask=o, opening=opening, solid=solid, bridges=n_bridges,
                        holes=holes, notches=notches))
    return res


def report(res):
    for r in res:
        s = r["solid"]
        core = s.buffer(-MIN_WEB / 2 * .9)
        parts = len(getattr(core, "geoms", [core]))
        open_area = r["opening"].area
        r.update(pieces=len(getattr(s, "geoms", [s])), core_parts=parts,
                 open_pct=100 * open_area / (MOTIF_W * MOTIF_H), open_area=open_area)
        print(r["id"], "Teile:", r["pieces"], "Kerne:", parts, "Stege:", r["bridges"],
              "Öffnung %.1f%%" % r["open_pct"], "Inseln in Öffnung:",
              sum(len(p.interiors) for p in getattr(r["opening"], "geoms", [r["opening"]])),
              "Vertices:", shapely.count_coordinates(s), "valid", s.is_valid)
