"""Erzeugt USDZ-Modelle, technische Zeichnung (PDF) und gesprühtes Bild."""
import os, pickle, subprocess, sys, tempfile
import cv2
import numpy as np
import shapely
from motiv import CREDIT, PAPER_NOTE, PREFIX, TITLE
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from stencil_geom import (LAYERS, MARGIN, MOTIF_H, MOTIF_W, NOTCH_D, NOTCH_PITCH, NOTCH_W, PLATE_H, PLATE_W,
                          PX_MM, REG_D, REG_HOLES, THICKNESS, MIN_WEB)

OUT = "ausgabe"
DATE = "2026-10-30"


def polys(g):
    return list(g.geoms) if hasattr(g, "geoms") else [g]


# ---------- USDZ ----------
def mesh(solid):
    """Extrudiert das Plattenpolygon zu einem geschlossenen Dreiecksnetz (m, Y oben, stehend)."""
    tris = shapely.get_parts(shapely.constrained_delaunay_triangles(solid))
    pts, idx = [], {}
    def v(x, y, z):
        k = (round(x, 3), round(y, 3), z)
        if k not in idx:
            idx[k] = len(pts)
            pts.append(((x - PLATE_W / 2) / 1000, (PLATE_H - y) / 1000, z / 1000))
        return idx[k]
    faces = []
    for t in tris:
        c = np.array(t.exterior.coords[:3])
        (ax, ay), (bx, by), (cx, cy) = c
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0:   # y-Spiegelung: im Modell gegen den Uhrzeigersinn
            c = c[::-1]
        a, b, d = [(float(p[0]), float(p[1])) for p in c]
        faces.append((v(*a, THICKNESS), v(*b, THICKNESS), v(*d, THICKNESS)))
        faces.append((v(*d, 0), v(*b, 0), v(*a, 0)))
    top = {e for f in faces[::2] for e in zip(f, f[1:] + f[:1])}
    for r in [solid.exterior, *solid.interiors]:
        c = [(float(x), float(y)) for x, y in r.coords]
        for a, b in zip(c[:-1], c[1:]):
            at, bt, ab, bb = v(*a, THICKNESS), v(*b, THICKNESS), v(*a, 0), v(*b, 0)
            if at == bt:
                continue
            if (at, bt) not in top:      # Seitenfläche gegenläufig zur Deckfläche
                at, bt, ab, bb = bt, at, bb, ab
            faces.append((bt, at, ab))
            faces.append((bt, ab, bb))
    return pts, faces


def write_usdz(r):
    pts, faces = mesh(r["solid"])
    col = "(%.4f, %.4f, %.4f)" % tuple((c / 255) ** 2.2 for c in r["rgb"])
    p_txt = ", ".join("(%.6f, %.6f, %.6f)" % p for p in pts)
    i_txt = ", ".join("%d, %d, %d" % f for f in faces)
    usda = f'''#usda 1.0
(
    defaultPrim = "Schablone"
    metersPerUnit = 1
    upAxis = "Y"
    doc = "Sprayschablone {r['id']} {r['name']}, {PLATE_W} x {PLATE_H} x {THICKNESS} mm"
)

def Xform "Schablone" (
    kind = "component"
)
{{
    def Mesh "Platte" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        uniform bool doubleSided = 0
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [{", ".join(["3"] * len(faces))}]
        int[] faceVertexIndices = [{i_txt}]
        point3f[] points = [{p_txt}]
        rel material:binding = </Schablone/Material>
    }}

    def Material "Material"
    {{
        token outputs:surface.connect = </Schablone/Material/PBR.outputs:surface>
        def Shader "PBR"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = {col}
            float inputs:roughness = 0.45
            float inputs:metallic = 0
            token outputs:surface
        }}
    }}
}}
'''
    with tempfile.TemporaryDirectory() as t:
        a, c = os.path.join(t, "a.usda"), os.path.join(t, "model.usdc")
        open(a, "w").write(usda)
        subprocess.run(["usdcat", a, "-o", c], check=True)
        dst = os.path.abspath(f"{OUT}/usdz/{r['file']}.usdz")
        if os.path.exists(dst):
            os.remove(dst)
        subprocess.run(["usdzip", dst, "model.usdc"], cwd=t, check=True, capture_output=True)
    r["tris"] = len(faces)
    # Dichtheit: jede Kante genau zweimal, gegenläufig
    e = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            e[(a, b)] = e.get((a, b), 0) + 1
    r["open_edges"] = sum(1 for (a, b) in e if (b, a) not in e)
    print(r["id"], "Dreiecke", len(faces), "offene Kanten", r["open_edges"], os.path.getsize(dst) // 1024, "KB")


# ---------- Spraybild ----------
def raster(geom, W, H, s, ox=0, oy=0):
    m = np.zeros((H, W), np.uint8)
    for p in polys(geom):
        cv2.fillPoly(m, [np.int32((np.array(p.exterior.coords) + (ox, oy)) * s * 16)], 255, cv2.LINE_AA, 4)
        for i in p.interiors:
            cv2.fillPoly(m, [np.int32((np.array(i.coords) + (ox, oy)) * s * 16)], 0, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255


def spray(res):
    rng = np.random.default_rng(1962)
    s = 6
    W, H = PLATE_W * s, PLATE_H * s
    fine = rng.normal(0, 1, (H, W)).astype(np.float32)
    paper = 1 + .012 * cv2.GaussianBlur(fine, (0, 0), 1.2) * 3 + .010 * cv2.GaussianBlur(fine, (0, 0), 7) * 12
    img = np.dstack([np.full((H, W), c, np.float32) for c in (232, 241, 246)]) * paper[..., None]   # BGR
    for r, (dx, dy) in zip(res, [(0, 0), (.25, -.15), (-.2, .2), (.15, .2)]):   # Passungsversatz
        m = raster(r["opening"], W, H, s, dx, dy)
        # Sprühkante: wenige Zehntel unscharf, leichter Unterdruck plus Sprühnebel
        edge = cv2.GaussianBlur(m, (0, 0), 1.3)
        halo = cv2.GaussianBlur(m, (0, 0), 5)
        n = rng.random((H, W), dtype=np.float32)
        speck = (n < .05 * halo * (1 - m)).astype(np.float32)
        density = .96 + .035 * cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 40) * 40
        a = np.clip(np.maximum(edge * density, speck * .75), 0, 1)[..., None]
        col = np.array(r["rgb"][::-1], np.float32)
        img = img * (1 - a) + col * a * (1 + .008 * cv2.GaussianBlur(fine, (0, 0), 1)[..., None] * 3)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUT}/{PREFIX}_spraybild.png", img)
    c = MARGIN * s
    cv2.imwrite(f"{OUT}/{PREFIX}_spraybild_motiv.png", img[c:-c, c:-c])


# ---------- Technische Zeichnung ----------
def path(c, geom, X, Y, k):
    p = c.beginPath()
    def ring(r):
        xy = np.array(r.coords)
        p.moveTo(X(xy[0][0]), Y(xy[0][1]))
        for x, y in xy[1:]:
            p.lineTo(X(x), Y(y))
        p.close()
    for g in polys(geom):
        ring(g.exterior)
        for i in g.interiors:
            ring(i)
    return p


def dim_h(c, x0, x1, y, yref, txt):
    c.setLineWidth(.18)
    c.line(x0, yref, x0, y + (1 if y > yref else -1) * mm)
    c.line(x1, yref, x1, y + (1 if y > yref else -1) * mm)
    c.line(x0, y, x1, y)
    for x, d in ((x0, 1), (x1, -1)):
        pp = c.beginPath(); pp.moveTo(x, y); pp.lineTo(x + d * 2.5 * mm, y + .6 * mm); pp.lineTo(x + d * 2.5 * mm, y - .6 * mm); pp.close()
        c.drawPath(pp, fill=1, stroke=0)
    c.setFont("Helvetica", 7.5)
    c.setFillColorRGB(1, 1, 1); w = c.stringWidth(txt, "Helvetica", 7.5)
    c.rect((x0 + x1) / 2 - w / 2 - 1, y - 3, w + 2, 6, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0); c.drawCentredString((x0 + x1) / 2, y - 2.6, txt)


def dim_v(c, y0, y1, x, xref, txt):
    c.saveState(); c.translate(x, 0); c.rotate(90)
    # gedreht: (u, v) = (y, -x+x_alt)
    dim_h(c, y0, y1, 0, -(xref - x), txt)
    c.restoreState()


def frame(c, sheet, sheets):
    W, H = landscape(A3)
    c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(.7)
    c.rect(10 * mm, 10 * mm, W - 20 * mm, H - 20 * mm)
    return W, H


def title_block(c, x, y, w, rows, sheet, sheets, title, scale):
    h = 7 * mm
    tot = h * (len(rows) + 2)
    c.setLineWidth(.5); c.rect(x, y, w, tot)
    c.setFont("Helvetica-Bold", 11); c.drawString(x + 2 * mm, y + tot - 7 * mm, title)
    c.setLineWidth(.18)
    c.line(x, y + tot - h * 1.0 - 3 * mm, x + w, y + tot - h * 1.0 - 3 * mm)
    yy = y + tot - h * 1.0 - 3 * mm
    all_rows = rows + [("Maßstab / Blatt", f"{scale}   /   {sheet} von {sheets}")]
    for k, v in all_rows:
        yy -= (tot - h * 1.0 - 3 * mm) / len(all_rows)
        c.line(x, yy, x + w, yy)
        c.setFont("Helvetica", 6.5); c.drawString(x + 2 * mm, yy + 2 * mm, k)
        c.setFont("Helvetica-Bold", 8); c.drawString(x + 34 * mm, yy + 2 * mm, v)


def sheet_plate(c, r, sheet, sheets):
    W, H = frame(c, sheet, sheets)
    k = .5 * mm
    ox, oy = 32 * mm, 268 * mm
    X = lambda x: ox + x * k
    Y = lambda y: oy - y * k
    tint = [v / 255 * .25 + .75 for v in r["rgb"]]
    c.setFillColorRGB(*tint); c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(.22)
    c.drawPath(path(c, r["solid"], X, Y, k), fill=1, stroke=1, fillMode=0)
    # Motivfeld und Mittellinien
    c.setDash([6, 2]); c.setLineWidth(.15)
    c.rect(X(MARGIN), Y(MARGIN + MOTIF_H), MOTIF_W * k, MOTIF_H * k)
    c.setDash([8, 2, 1, 2])
    c.line(X(PLATE_W / 2), Y(-6), X(PLATE_W / 2), Y(PLATE_H + 6)); c.line(X(-6), Y(PLATE_H / 2), X(PLATE_W + 6), Y(PLATE_H / 2))
    c.setDash()
    c.setFillColorRGB(0, 0, 0)
    dim_h(c, X(0), X(PLATE_W), Y(PLATE_H) - 12 * mm, Y(PLATE_H), f"{PLATE_W}")
    dim_h(c, X(MARGIN), X(MARGIN + MOTIF_W), Y(0) + 8 * mm, Y(MARGIN), f"{MOTIF_W}  (Motivfeld)")
    dim_v(c, Y(PLATE_H), Y(0), X(0) - 13 * mm, X(0), f"{PLATE_H}")
    dim_v(c, Y(MARGIN + MOTIF_H), Y(MARGIN), X(PLATE_W) + 8 * mm, X(PLATE_W - MARGIN), f"{MOTIF_H}")

    # Passlochtabelle
    rx = 268 * mm
    c.setFont("Helvetica-Bold", 8.5); c.drawString(rx, 276 * mm, "Passlöcher (Ursprung: Ecke oben links)")
    c.setFont("Helvetica", 7.5)
    for i, (x, y) in enumerate(REG_HOLES):
        c.drawString(rx, (270 - i * 4.6) * mm, f"P{i + 1}   Ø{REG_D:.0f} H9      x = {x}      y = {y}")
    c.drawString(rx, 251 * mm, "P4 bewusst versetzt: Platte kann nicht gewendet aufgelegt werden.")
    c.setFont("Helvetica-Bold", 8.5); c.drawString(rx, 243 * mm, "Kennkerben Oberkante")
    c.setFont("Helvetica", 7.5)
    c.drawString(rx, 238 * mm, f"{r['no']} × {NOTCH_W} × {NOTCH_D} mm, Teilung {NOTCH_PITCH} mm, mittig zur Plattenachse")

    # Schnitt
    c.setFont("Helvetica-Bold", 8.5); c.drawString(rx, 229 * mm, "Schnitt durch Platte (5:1)")
    sy, t = 218 * mm, THICKNESS * 5 * mm
    c.setFillColorRGB(*tint); c.setLineWidth(.4)
    c.rect(rx, sy, 40 * mm, t, fill=1); c.rect(rx + 55 * mm, sy, 40 * mm, t, fill=1)
    c.setFillColorRGB(0, 0, 0); dim_h(c, rx + 40 * mm, rx + 55 * mm, sy + 10 * mm, sy + t, "Öffnung ≥ 2")
    c.setFont("Helvetica", 7.5); c.drawString(rx + 97 * mm, sy + 1.2 * mm, f"t = {THICKNESS:.1f}")

    notes = [
        "Hinweise",
        "1. Allgemeintoleranz ISO 2768-m, Passlöcher ±0,1 mm.",
        "2. Innenkonturen als Laserschnitt, Bahnkorrektur halbe Fugenbreite nach Innen.",
        f"3. Kleinste Materialbreite {MIN_WEB:.1f} mm, geprüft über Negativversatz der Kontur.",
        "4. Schablone ist einteilig: alle Inseln über Stege mit dem Rahmen verbunden.",
        "5. Stege liegen wo möglich unter der nächsten, dunkleren Sprühebene.",
        "6. Auf Passstifte Ø6 setzen, mit Repositionierkleber plan fixieren, senkrecht sprühen.",
        "7. Vor dem Ablösen 5 min antrocknen lassen. Reihenfolge hell nach dunkel.",
    ]
    y = 203 * mm
    for i, n in enumerate(notes):
        c.setFont("Helvetica-Bold" if i == 0 else "Helvetica", 8.5 if i == 0 else 7.5)
        c.drawString(rx, y, n); y -= 4.6 * mm

    # Legende
    c.setFillColorRGB(*[v / 255 for v in r["rgb"]]); c.setLineWidth(.3)
    c.rect(rx, 156 * mm, 14 * mm, 9 * mm, fill=1)
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica", 7.5)
    c.drawString(rx + 17 * mm, 161 * mm, f"Sprühfarbe {r['no']}: {r['name']}, {r['ral']}")
    c.drawString(rx + 17 * mm, 157 * mm, f"sRGB {r['rgb'][0]}, {r['rgb'][1]}, {r['rgb'][2]}  (Modellfarbe in USDZ)")

    rows = [
        ("Benennung", f"Sprayschablone {r['id']} – {r['name']}"),
        ("Motiv", f"{TITLE}, Vierfarb-Spraydruck"),
        ("Werkstoff", f"PETG-Folie, t = {THICKNESS:.1f} mm"),
        ("Abmessung", f"{PLATE_W} × {PLATE_H} × {THICKNESS:.0f} mm"),
        ("Öffnungsfläche", f"{r['open_area'] / 100:.0f} cm²  ({r['open_pct']:.1f} % des Motivfelds)"),
        ("Stege", f"{r['bridges']} geroutet"),
        ("Datei 3D", f"{r['file']}.usdz"),
        ("Datum", DATE),
    ]
    title_block(c, rx, 16 * mm, 140 * mm, rows, sheet, sheets, f"{PREFIX}-{r['id']}", "1:2")
    c.showPage()


def sheet_overview(c, res, sheets):
    W, H = frame(c, 1, sheets)
    c.setFont("Helvetica-Bold", 13); c.drawString(18 * mm, 272 * mm, "Sprayfolge und Passung")
    k = .165 * mm
    for i, r in enumerate(res):
        ox, oy = (20 + i * 76) * mm, 262 * mm
        X = lambda x: ox + x * k
        Y = lambda y: oy - y * k
        c.setFillColorRGB(*[v / 255 for v in r["rgb"]]); c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(.12)
        c.drawPath(path(c, r["solid"], X, Y, k), fill=1, stroke=1, fillMode=0)
        c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 9)
        c.drawString(ox, oy - (PLATE_H * k) - 6 * mm, f"{r['no']}. {r['id']} {r['name']}")
        c.setFont("Helvetica", 7.5)
        c.drawString(ox, oy - (PLATE_H * k) - 10 * mm, f"{r['ral']}  ·  {r['open_pct']:.1f} %  ·  {r['bridges']} Stege")
    im = f"{OUT}/{PREFIX}_spraybild.png"
    small = cv2.resize(cv2.imread(im), None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    tmp = os.path.join(tempfile.gettempdir(), f"{PREFIX}_small.jpg"); cv2.imwrite(tmp, small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    c.drawImage(tmp, 325 * mm, 134 * mm, 84 * mm, 84 * PLATE_H / PLATE_W * mm)
    c.setFont("Helvetica", 7.5); c.drawString(325 * mm, 130 * mm, "Simulation aus den Schnittkonturen")
    c.setFont("Helvetica-Bold", 9); c.drawString(18 * mm, 118 * mm, "Ablauf")
    steps = [
        "1. Träger plan auflegen, Passstifte P1 bis P4 setzen (Bohrbild siehe Blatt 2).",
        "2. S1 bis S4 in dieser Reihenfolge auf die Stifte legen, jeweils ganzflächig dünn sprühen, Platte vor der nächsten Ebene abnehmen.",
        "3. Ebenen sind kumulativ: jede dunklere Ebene liegt vollständig auf der vorherigen, damit bei Versatz keine Blitzer entstehen.",
        PAPER_NOTE,
        "5. Haftfarbe mit hoher Deckkraft verwenden, sonst scheint Gelb unter Rosa durch.",
    ]
    y = 112 * mm; c.setFont("Helvetica", 8)
    for s in steps:
        c.drawString(18 * mm, y, s); y -= 5 * mm
    c.setFont("Helvetica", 7)
    c.drawString(18 * mm, 80 * mm, CREDIT)
    rows = [("Benennung", "Schablonensatz, 4-teilig"), ("Motiv", f"{TITLE}, Vierfarb-Spraydruck"),
            ("Motivfeld", f"{MOTIF_W} × {MOTIF_H} mm"), ("Datum", DATE)]
    title_block(c, 269 * mm, 16 * mm, 140 * mm, rows, 1, sheets, f"{PREFIX}-SATZ", "ohne")
    c.showPage()


def main():
    res = pickle.load(open(sys.argv[1], "rb"))
    for r in res:
        write_usdz(r)
    spray(res)
    c = canvas.Canvas(f"{OUT}/{PREFIX}_technische_zeichnung.pdf", pagesize=landscape(A3))
    c.setTitle(f"Sprayschablonen {TITLE}"); c.setPageCompression(1)
    sheet_overview(c, res, 5)
    for i, r in enumerate(res):
        sheet_plate(c, r, i + 2, 5)
    c.save()


if __name__ == "__main__":
    main()
