"""Tonwerttrennung: Foto -> vier kumulative Sprühebenen (Rasterstufe)."""
import cv2
import numpy as np

SRC = "quelle/Brigitte_Bardot_-_1962.jpg"
PX_MM = 6                 # Arbeitsraster
MOTIF_W, MOTIF_H = 360, 450   # mm
CROP_H = 881              # Quellbild-Pixel, Seitenverhältnis 4:5

# Umrisse in Quellbild-Pixeln (705 x 900)
FIGURE = [(150,0),(118,55),(92,120),(76,200),(84,285),(100,350),(108,420),(112,470),(110,520),(128,590),
          (158,655),(200,692),(246,706),(252,770),(262,830),(240,881),(705,881),(690,820),(672,740),(660,640),
          (645,540),(632,450),(645,370),(662,290),(664,190),(645,95),(615,0)]
FACE = [(150,345),(190,320),(240,300),(290,345),(330,365),(390,395),(430,440),(462,490),(482,545),(470,600),
        (440,640),(400,680),(350,735),(330,790),(300,850),(262,881),(262,830),(252,770),(246,706),(200,692),
        (158,655),(128,590),(110,520),(112,470),(120,410)]
LIPS = [(155,552),(185,546),(215,552),(250,562),(298,582),(275,600),(235,618),(195,618),(170,595)]
TEETH = [(165,560),(205,562),(245,570),(285,578),(265,592),(225,596),(185,594)]


def poly_mask(points, shape, scale):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.array(points, float) * scale)], 255)
    return m


def soft(mask, sigma):
    return cv2.GaussianBlur(mask, (0, 0), sigma) > 127


def separate():
    """Liefert dict ebene -> bool-Maske (True = Öffnung) und Graustufenbild."""
    W, H = MOTIF_W * PX_MM, MOTIF_H * PX_MM
    img = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)[:CROP_H]
    g = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6)).apply(g)
    smooth = lambda x: cv2.GaussianBlur(cv2.bilateralFilter(x, 0, 22, 14), (0, 0), 5).astype(np.float32) / 255
    g, gl = smooth(g), smooth(clahe)
    scale = W / img.shape[1]
    poly = poly_mask(FIGURE, (H, W), scale)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    # Haarkontur oben/rechts folgt dem Foto, unten links dem Polygon
    hair_zone = np.zeros((H, W), bool)
    hair_zone[: int(430 * scale)] = True
    hair_zone[:, int(420 * scale):] = True
    fig = cv2.erode(poly, k(24)) > 0
    fig |= (cv2.dilate(poly, k(30)) > 0) & (g > 0.60) & hair_zone
    fig |= (poly > 0) & ~hair_zone
    fig = soft(fig.astype(np.uint8) * 255, 12)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fig.astype(np.uint8))
    fig = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    face = soft(poly_mask(FACE, (H, W), scale), 14) & fig
    lips = soft(poly_mask(LIPS, (H, W), scale), 6) & face
    hair = fig & ~face

    gf = 0.72 * g + 0.28 * gl
    teeth = lips & (g > 0.56) & soft(poly_mask(TEETH, (H, W), scale), 4)
    yellow = fig & ~teeth & ((hair & (g < 0.93)) | (face & (gf < 0.88)))
    rose = (face & ~teeth & ((gf < 0.60) | lips)) | (hair & (g < 0.50))
    petrol = ~fig | (hair & (g < 0.28)) | (face & (gf < 0.27))
    black = fig & ((g < 0.15) | (face & (g < 0.25)))
    return dict(yellow=yellow, rose=rose, petrol=petrol, black=black), g


if __name__ == "__main__":
    m, g = separate()
    pal = dict(yellow=(40,205,245), rose=(120,110,235), petrol=(100,85,10), black=(30,25,20))
    out = np.full(g.shape + (3,), 0, np.uint8); out[:] = (225,240,247)
    for k in ("yellow","rose","petrol","black"):
        out[m[k]] = pal[k]
    cv2.imwrite("/private/tmp/claude-501/-Users-haraldbeker-Schablone/902e6aa7-2c7f-478b-aa57-5677d0131062/scratchpad/prev.png",
                cv2.resize(out, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA))
