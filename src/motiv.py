"""Motivauswahl über Umgebungsvariable MOTIV (bb oder maria)."""
import os

NAME = os.environ.get("MOTIV", "bb")
if NAME == "maria":
    from separation_maria import separate, PX_MM, MOTIF_W, MOTIF_H
    PREFIX, TITLE = "Maria", "Maria"
    PAPER_NOTE = "4. Die Figur liegt vollständig unter S1, es bleibt kein Papierton stehen. Hintergrund entsteht allein durch S3."
    CREDIT = "Vorlage: privates Foto (quelle/Maria.png), Namensetikett und Kopfhörerkabel retuschiert."
else:
    from separation import separate, PX_MM, MOTIF_W, MOTIF_H
    PREFIX, TITLE = "BB", "Brigitte Bardot"
    PAPER_NOTE = "4. Papierton (Zähne, Lichtpunkte) bleibt unbesprüht. Hintergrund entsteht allein durch S3."
    CREDIT = "Vorlage: Werbefoto „Brigitte Bardot - 1962“, Wikimedia Commons, als gemeinfrei gekennzeichnet (Quelle: MGM)."
