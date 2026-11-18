"""Tonwerttrennung für beliebige Fotos: automatisch freistellen, Stufen über Perzentile."""
import cv2
import numpy as np

PX_MM = 6                 # Arbeitsraster
MOTIF_W, MOTIF_H = 360, 450   # mm

# Anteil der Figurfläche je Ebene: rose, petrol, black zählen die dunkelsten Pixel,
# white die hellsten. Weiß hat keine eigene Schablone: dort sind alle vier Platten geschlossen.
SHARE = dict(white=0.10, rose=0.50, petrol=0.20, black=0.07)


def crop45(img):
    """Mittiger Zuschnitt auf 4:5."""
    h, w = img.shape[:2]
    if w / h > 4 / 5:
        cw = round(h * 4 / 5)
        x = (w - cw) // 2
        return img[:, x:x + cw]
    ch = round(w * 5 / 4)
    y = max(0, (h - ch) // 3)        # eher oben anschneiden: Köpfe sitzen meist im oberen Drittel
    return img[y:y + ch]


def figure(img):
    """Hauptmotiv per GrabCut, Startrechteck mit 6 % Rand, unten offen (Brustbild)."""
    h, w = img.shape[:2]
    s = 600 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    mx, my = round(sw * .06), round(sh * .06)
    gc = np.zeros((sh, sw), np.uint8)
    cv2.grabCut(small, gc, (mx, my, sw - 2 * mx, sh - my), np.zeros((1, 65)), np.zeros((1, 65)), 5,
                cv2.GC_INIT_WITH_RECT)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def soft(mask, sigma):
    m = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
    return cv2.GaussianBlur(m, (0, 0), sigma) > 127


def fill_holes(m):
    h, w = m.shape
    inv = np.pad(~m, 1, constant_values=True).astype(np.uint8)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)   # Außenfläche löschen, Löcher bleiben
    return m | (inv[1:-1, 1:-1] > 0)


def separate_image(img, cutout=True, share=None):
    """img: BGR-Foto. Liefert dict ebene -> bool-Maske (True = Öffnung), Graustufenbild, Zuschnitt."""
    share = {**SHARE, **(share or {})}
    W, H = MOTIF_W * PX_MM, MOTIF_H * PX_MM
    crop = crop45(img)
    up = lambda c: cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
    smooth = lambda x: cv2.GaussianBlur(cv2.bilateralFilter(x, 0, 22, 14), (0, 0), 5).astype(np.float32) / 255
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6)).apply(gray)
    g = 0.72 * smooth(up(gray)) + 0.28 * smooth(up(clahe))

    if cutout:
        fig = soft(up(figure(crop)), 10)
        n, lab, st, _ = cv2.connectedComponentsWithStats(fig.astype(np.uint8))
        fig = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]) if n > 1 else np.ones((H, W), bool)
        fig = fill_holes(fig)
    else:
        fig = np.ones((H, W), bool)

    q = lambda p: np.quantile(g[fig], p)
    yellow = fig & (g < q(1 - share["white"])) if share["white"] > 0 else fig.copy()
    rose = fig & (g < q(share["rose"]))
    petrol = ~fig | (fig & (g < q(share["petrol"])))
    black = fig & (g < q(share["black"]))
    return dict(yellow=yellow, rose=rose, petrol=petrol, black=black), g, crop
