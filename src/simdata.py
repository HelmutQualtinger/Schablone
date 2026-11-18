"""Schablonensatz -> DATA-Objekt der Simulationsseite (Masken als PNG, Konturen in Bühnenpixeln)."""
import base64
import cv2
import numpy as np
from stencil_geom import MOTIF_W, MOTIF_H

S = 2                              # Bühnenpixel pro mm (Bühne 840 x 1020 = Platte 420 x 510 mm)
MW, MH = MOTIF_W * S, MOTIF_H * S


def b64(ext, img, *args):
    ok, buf = cv2.imencode(ext, img, list(args))
    return base64.b64encode(buf).decode()


def small(mask):
    """Bool-Maske im Arbeitsraster -> 8-bit-Maske in Bühnengröße."""
    return cv2.resize(mask.astype(np.uint8) * 255, (MW, MH), interpolation=cv2.INTER_AREA)


def islands(raw):
    """Material, das ohne Stege keine Verbindung zum Rahmen hätte."""
    solid = np.pad(small(raw) < 128, 1, constant_values=True).astype(np.uint8)
    n, lab = cv2.connectedComponents(solid, connectivity=4)
    return ((lab != lab[0, 0]) & (solid > 0))[1:-1, 1:-1].astype(np.uint8) * 255


def rings(geom):
    out = []
    for p in getattr(geom, "geoms", [geom]):
        p = p.simplify(.4)
        for r in [p.exterior, *p.interiors]:
            c = np.round(np.array(r.coords[:-1]) * S).astype(int).ravel().tolist()
            if len(c) >= 6:
                out.append(c)
    return out


def simdata(res, gray, crop):
    bridges = [r["bridges"] for r in res]
    k = int(np.argmax(bridges))
    raw, final = small(res[k]["raw"]), small(res[k]["mask"])
    photo = cv2.resize(crop, (MW, MH), interpolation=cv2.INTER_AREA)
    g = cv2.resize(np.clip(gray * 255, 0, 255).astype(np.uint8), (MW, MH), interpolation=cv2.INTER_AREA)
    png = lambda m: b64(".png", m, cv2.IMWRITE_PNG_COMPRESSION, 9)
    return dict(
        K=k, bridges=bridges,
        open=[round(100 * r["opening"].area / (MOTIF_W * MOTIF_H), 1) for r in res],
        masks=[png(small(r["mask"])) for r in res],
        raw=png(raw), isl=png(islands(res[k]["raw"])),
        br=png(((raw > 127) & (final < 128)).astype(np.uint8) * 255),
        gray=b64(".jpg", g, cv2.IMWRITE_JPEG_QUALITY, 82),
        photo=b64(".jpg", photo, cv2.IMWRITE_JPEG_QUALITY, 85),
        rings=[rings(r["opening"]) for r in res],
    )
