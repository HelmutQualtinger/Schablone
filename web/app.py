"""Webseite: Foto hochladen, Schablonensatz berechnen, Ablauf als Simulation zeigen.

Start aus dem Projektordner:  python3 web/app.py   ->  http://127.0.0.1:8000
"""
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from separation_auto import separate_image   # noqa: E402
from druck import bild, stl                     # noqa: E402
from simdata import simdata                    # noqa: E402
from stencil_geom import build                 # noqa: E402

MAX_MB = 20
MAX_SIDE = 3000          # längere Bildseite, größer wird verkleinert
KEEP_S = 30 * 60         # Druckdaten so lange vorhalten
FILES = os.path.join(tempfile.gettempdir(), "schablone_web")   # übersteht einen Neustart des Servers
os.makedirs(FILES, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
pool = ThreadPoolExecutor(max_workers=1)       # rechenintensiv: Aufträge nacheinander
jobs, lock = {}, threading.Lock()


def update(jid, **kw):
    with lock:
        jobs[jid].update(kw, t=time.time())


def run(jid, img, cutout, share):
    try:
        update(jid, schritt="Tonwerte trennen …", anteil=.08)
        m, g, crop = separate_image(img, cutout=cutout, share=share)
        progress = lambda no: update(jid, schritt=f"Platte {no} von 4: Stege und Konturen …", anteil=.3 + .15 * (no - 1))
        res = build(m, progress)
        update(jid, schritt="Simulation und Druckdaten vorbereiten …", anteil=.95)
        data = simdata(res, g, crop)
        stl_bytes, scale = stl(res)
        data["druck"] = dict(massstab=round(scale, 4), platte=[round(420 * scale, 1), round(510 * scale, 1)])
        for kind, body in (("stl", stl_bytes), ("png", bild(res))):
            with open(os.path.join(FILES, f"{jid}.{kind}"), "wb") as fh:
                fh.write(body)
        update(jid, data=data, fertig=True, anteil=1, schritt="Fertig")
    except Exception:
        traceback.print_exc()
        update(jid, fehler="Die Berechnung ist fehlgeschlagen. Ein anderes Foto oder andere Anteile versuchen.")


def purge():
    now = time.time()
    with lock:
        for jid in [j for j, v in jobs.items() if now - v["t"] > KEEP_S]:
            del jobs[jid]
    for name in os.listdir(FILES):
        path = os.path.join(FILES, name)
        if now - os.path.getmtime(path) > KEEP_S:
            os.remove(path)


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "web", "static"), "index.html")


@app.get("/og.jpg")
def og_image():
    return send_from_directory(os.path.join(ROOT, "web", "static"), "og.jpg", max_age=3600)


@app.post("/api/auftrag")
def create():
    purge()
    f = request.files.get("foto")
    if not f:
        return jsonify(fehler="Kein Foto übertragen."), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(fehler="Das Bild lässt sich nicht lesen. JPEG oder PNG verwenden."), 400
    h, w = img.shape[:2]
    if min(h, w) < 200:
        return jsonify(fehler="Das Foto ist zu klein, mindestens 200 Pixel Kantenlänge."), 400
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def pct(name, lo, hi, default):
        try:
            return min(hi, max(lo, float(request.form.get(name, default)))) / 100
        except ValueError:
            return default / 100
    share = dict(white=pct("white", 0, 40, 10), rose=pct("rose", 20, 80, 50), petrol=pct("petrol", 5, 45, 20),
                 black=pct("black", 1, 25, 7))
    # Ebenen sind kumulativ: dunklere Ebene darf nicht größer sein als die hellere, Weiß nicht in Rosa ragen
    share["rose"] = min(share["rose"], 1 - share["white"])
    share["petrol"] = min(share["petrol"], share["rose"])
    share["black"] = min(share["black"], share["petrol"])
    cutout = request.form.get("freistellen", "1") == "1"

    jid = uuid.uuid4().hex
    with lock:
        waiting = sum(1 for v in jobs.values() if "fertig" not in v and "fehler" not in v)
        jobs[jid] = dict(t=time.time(), anteil=0, schritt="Wartet auf freien Rechenplatz …" if waiting else "Startet …")
    pool.submit(run, jid, img, cutout, share)
    return jsonify(id=jid), 202


@app.get("/api/auftrag/<jid>")
def state(jid):
    with lock:
        j = jobs.get(jid)
        if j is None:
            return jsonify(fehler="Auftrag unbekannt oder abgelaufen."), 404
        out = {k: j[k] for k in ("schritt", "anteil", "fehler", "data") if k in j}
        j.pop("data", None)        # Simulationsdaten nur einmal ausliefern, Downloads bleiben KEEP_S verfügbar
    return jsonify(out)


DOWNLOADS = dict(stl="model/stl", png="image/png")


@app.get("/api/auftrag/<jid>/<kind>")
def download(jid, kind):
    if kind not in DOWNLOADS:
        return jsonify(fehler="Unbekannte Datei."), 404
    path = os.path.join(FILES, f"{jid}.{kind}")
    if not (len(jid) == 32 and all(c in "0123456789abcdef" for c in jid) and os.path.isfile(path)):
        return jsonify(fehler="Die Datei ist abgelaufen. Das Foto bitte neu berechnen lassen."), 404
    with open(path, "rb") as fh:
        body = fh.read()
    return Response(body, mimetype=DOWNLOADS[kind], headers={"Content-Disposition": "attachment"})   # Name setzt die Seite


@app.errorhandler(413)
def too_large(_):
    return jsonify(fehler=f"Das Foto ist größer als {MAX_MB} MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), threaded=True)
