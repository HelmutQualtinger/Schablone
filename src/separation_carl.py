"""Tonwerttrennung Carl: Foto -> vier kumulative Sprühebenen (Rasterstufe)."""
import cv2
import numpy as np

SRC = "quelle/Carl2.JPG"
PX_MM = 6                 # Arbeitsraster
MOTIF_W, MOTIF_H = 360, 450   # mm
CROP_Y, CROP_H = 130, 1355    # Quellbild-Pixel, Seitenverhältnis 4:5

# Umrisse in Quellbild-Pixeln (1084 x 1568), notiert in halber Auflösung
_h = lambda P: [(2 * x, 2 * y) for x, y in P]
SKIN = _h([(60, 330), (90, 290), (150, 255), (250, 235), (350, 240), (430, 265), (470, 320), (490, 360), (518, 362),
           (525, 420), (505, 500), (478, 530), (465, 620), (430, 690), (400, 730), (150, 730), (100, 690), (62, 600),
           (45, 520), (15, 485), (8, 380), (30, 345)])
MOUTH = _h([(190, 595), (240, 585), (290, 590), (340, 588), (392, 595), (360, 645), (300, 672), (240, 668), (205, 632)])
EYES = _h([(110, 400), (490, 400), (490, 470), (110, 470)])       # Brillengläser: Pupillen dürfen schwarz werden
GLASSES = _h([(85, 375), (508, 375), (508, 495), (85, 495)])
SHIRT_Y = 1340                                                    # Shirt nur unterhalb davon

SKIN_R = 0.80        # Rotkanal: Haut hell, Haar dunkler
TEETH_G = 0.62


def poly_mask(points, shape, scale):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32((np.array(points, float) - (0, CROP_Y)) * scale)], 255)
    return m


def soft(mask, sigma):
    return cv2.GaussianBlur(mask.astype(np.uint8) * 255 if mask.dtype == bool else mask, (0, 0), sigma) > 127


def fill_holes(m):
    h, w = m.shape
    inv = np.pad(~m, 1).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    return m | (inv[1:-1, 1:-1] > 0)


def figure(img):
    """Freistellen vor hellgrauem Hintergrund: unbunte helle Randflächen als Start, GrabCut verfeinert."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 2), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    n, lab_ = cv2.connectedComponents(((L > 215) & (A < 1.5) & (np.abs(B) < 5)).astype(np.uint8))
    border = set(lab_[0]) | set(lab_[-1]) | set(lab_[:, 0]) | set(lab_[:, -1])
    border.discard(0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(np.isin(lab_, list(border)).astype(np.uint8), cv2.MORPH_OPEN, k)
    gc = np.where(bg > 0, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype(np.uint8)
    gc[cv2.erode(bg, k, iterations=3) > 0] = cv2.GC_BGD
    gc[cv2.erode(1 - bg, k, iterations=4) > 0] = cv2.GC_FGD
    cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    return ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255


def source():
    """Foto auf 4:5 beschnitten und Figurmaske."""
    img = cv2.imread(SRC, cv2.IMREAD_COLOR)
    return img[CROP_Y:CROP_Y + CROP_H], figure(img)[CROP_Y:CROP_Y + CROP_H]


def separate():
    """Liefert dict ebene -> bool-Maske (True = Öffnung) und Graustufenbild."""
    W, H = MOTIF_W * PX_MM, MOTIF_H * PX_MM
    crop, alpha = source()
    up = lambda c: cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
    smooth = lambda x, s=5: cv2.GaussianBlur(cv2.bilateralFilter(x, 0, 22, 14), (0, 0), s).astype(np.float32) / 255
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = smooth(up(gray))
    r = smooth(up(crop[..., 2]))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = cv2.GaussianBlur(up(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)), (0, 0), 8)
    scale = W / crop.shape[1]
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    fig = soft(up(alpha), 10)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(fig.astype(np.uint8))
    fig = lab_ == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    # Haaransatz folgt dem Foto, das Polygon begrenzt nur grob
    zone = poly_mask(SKIN, (H, W), scale) > 0
    skin = soft(zone & (r > SKIN_R), 8) & fig
    # Die Brille teilt die Haut in Stirn, Wangen und Ohren: alle großen Stücke behalten
    n, lab_, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8))
    keep = st[:, cv2.CC_STAT_AREA] >= 300 * PX_MM ** 2
    keep[0] = False
    skin = cv2.morphologyEx(keep[lab_].astype(np.uint8), cv2.MORPH_CLOSE, k(30)) > 0
    skin = fill_holes(skin) & zone & fig
    # Shirt: unbunte Fläche unterhalb des Kinns
    low = np.zeros((H, W), bool)
    low[int((SHIRT_Y - CROP_Y) * scale):] = True
    shirt = soft(fig & low & (chroma < 9), 10) & ~skin
    hair = fig & ~skin & ~shirt
    mouth = poly_mask(MOUTH, (H, W), scale) > 0
    teeth = soft(mouth & (g > TEETH_G), 4)
    eyes = poly_mask(EYES, (H, W), scale) > 0
    # Brillenfassung ist nur 3 mm breit: aus dem kaum geglätteten Bild, auf Mindestbreite aufgedickt
    g0 = cv2.GaussianBlur(up(gray), (0, 0), 2).astype(np.float32) / 255
    frame = (poly_mask(GLASSES, (H, W), scale) > 0) & (g0 < 0.25)
    frame = soft(cv2.dilate(frame.astype(np.uint8) * 255, k(3)), 3) & fig
    gs = cv2.GaussianBlur(g, (0, 0), 30)    # großflächige Modellierung des Gesichts

    face = skin & ~teeth
    yellow = fig.copy()
    rose = (face & ((g < 0.62) | (gs < 0.72) | mouth)) | hair
    # Shirt bleibt gelb, nur die Streifen werden petrol
    petrol = ~fig | (hair & (g < 0.30)) | (face & (g < 0.38)) | (shirt & (g < 0.45))
    black = fig & ((hair & (g < 0.12)) | (face & (g < 0.18)) | (eyes & (g < 0.25)) | frame)
    return dict(yellow=yellow, rose=rose, petrol=petrol, black=black), g
