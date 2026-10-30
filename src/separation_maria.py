"""Tonwerttrennung Maria: Foto -> vier kumulative Sprühebenen (Rasterstufe)."""
import cv2
import numpy as np

SRC = "quelle/Maria.png"
PX_MM = 6                 # Arbeitsraster
MOTIF_W, MOTIF_H = 360, 450   # mm
CROP_X, CROP_W = 15, 829  # Quellbild-Pixel, Seitenverhältnis 4:5

# Retusche und Umrisse in Quellbild-Pixeln (860 x 1036)
LABEL = (355, 748, 525, 800)
CABLES = [[(232, 565), (262, 645), (290, 720), (320, 830), (345, 1000), (350, 1036)],
          [(678, 735), (660, 830), (632, 930), (585, 1036)]]
REMOTE = [(655, 825), (680, 830), (652, 915), (630, 910)]
FIGURE = [(0,905),(60,878),(130,862),(190,868),(150,835),(105,790),(70,715),(50,640),(60,530),(90,420),(130,300),
          (180,190),(240,110),(320,62),(400,55),(480,85),(560,145),(640,215),(700,295),(750,395),(790,490),(812,590),
          (805,675),(770,725),(700,755),(685,790),(800,838),(860,878),(860,1036),(0,1036)]
SKIN = [(218,450),(232,300),(265,215),(320,140),(400,170),(500,225),(590,285),(640,360),(672,430),(692,520),(686,620),
        (650,700),(668,770),(690,800),(800,840),(690,940),(600,1036),(345,1036),(262,890),(355,800),(300,690),(250,610),(222,530)]
SHIRT = [(0,905),(60,878),(130,862),(240,885),(345,1036),(0,1036)]
SHIRT_R = [(800,840),(860,878),(860,1036),(600,1036),(690,940)]
A_LIP = 0.60
LIPS = [(405,672),(470,655),(530,645),(585,620),(575,650),(530,685),(475,698),(430,690)]


def poly_mask(points, shape, scale):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32((np.array(points, float) - (CROP_X, 0)) * scale)], 255)
    return m


def soft(mask, sigma):
    return cv2.GaussianBlur(mask, (0, 0), sigma) > 127


def retouch(img):
    m = np.zeros(img.shape[:2], np.uint8)
    x0, y0, x1, y1 = LABEL
    m[y0:y1, x0:x1] = 255
    for c in CABLES:
        cv2.polylines(m, [np.int32(c)], False, 255, 16)
    cv2.fillPoly(m, [np.int32(REMOTE)], 255)
    return cv2.inpaint(img, m, 9, cv2.INPAINT_TELEA)


def separate():
    """Liefert dict ebene -> bool-Maske (True = Öffnung) und Graustufenbild."""
    W, H = MOTIF_W * PX_MM, MOTIF_H * PX_MM
    img = retouch(cv2.imread(SRC, cv2.IMREAD_COLOR))
    crop = img[:, CROP_X:CROP_X + CROP_W]
    up = lambda c: cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
    smooth = lambda x, s=5: cv2.GaussianBlur(cv2.bilateralFilter(x, 0, 22, 14), (0, 0), s).astype(np.float32) / 255
    # Rotkanal für Haut: Rötungen und Sommersprossen fallen weg, Augen und Brauen bleiben
    g = smooth(up(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
    r = smooth(up(crop[..., 2]))
    rl = smooth(cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6)).apply(up(crop[..., 2])))
    a = smooth(up(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 1]), 3)
    scale = W / CROP_W
    k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    fig = soft(poly_mask(FIGURE, (H, W), scale), 12)
    # Haaransatz und Wange folgen dem Foto, das Polygon begrenzt nur grob
    zone = cv2.dilate(poly_mask(SKIN, (H, W), scale), k(40)) > 0
    skin = soft(((zone & (r > 0.50)).astype(np.uint8) * 255), 8) & fig
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8))
    skin = lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, k(30)) > 0
    shirt = soft(poly_mask(SHIRT, (H, W), scale) | poly_mask(SHIRT_R, (H, W), scale), 6) & fig & ~skin
    lips = soft(poly_mask(LIPS, (H, W), scale), 5) & (a > A_LIP)
    hair = fig & ~skin & ~shirt
    gs = cv2.GaussianBlur(g, (0, 0), 14)

    rf = 0.6 * r + 0.4 * rl
    mouth = lips & (g < np.percentile(g[lips], 14))
    yellow = fig.copy()
    rose = (skin & ((rf < 0.665) | lips)) | hair | (shirt & (gs < 0.54))
    petrol = ~fig | (hair & (r < 0.42)) | (skin & (rf < 0.46)) | mouth | (shirt & (gs < 0.30))
    # Lichtinseln im Haar bräuchten Stege quer durchs Schwarz: nur große behalten
    n, lab, st, _ = cv2.connectedComponentsWithStats((hair & (r >= 0.27)).astype(np.uint8))
    keep = st[:, cv2.CC_STAT_AREA] >= 2000 * PX_MM ** 2
    keep[0] = False
    dark = hair & ~keep[lab]
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    eyes = np.zeros((H, W), bool)   # Pupillen liegen als Inseln außerhalb der Hautmaske
    eyes[: int(700 * scale)] = True
    black = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])) | (dark & eyes) | (skin & (r < 0.34))
    return dict(yellow=yellow, rose=rose, petrol=petrol, black=black), g
