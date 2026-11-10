"""
Einheitliche Überschrift für alle Grafiken (Klima im Wandel und DWD-Grafiken).

Drei Zeilen an festen Stellen im Bild - unabhängig vom Diagramm, von der Bildgröße und davon,
ob die Grafik auf dem PC oder auf GitHub entsteht:

    Temperaturmittel                         Zeile 1, fett
    Global | Jahr | ERA5                     Zeile 2, fett
    Abweichung von ... (1850-1900)           Zeile 3, normal

Größen und Abstände sind Anteile der Bildhöhe (abgemessen an der DWD-Monatsbilanz),
deshalb sieht es bei 16 x 9 und 20 x 11,25 Zoll gleich aus.

Verwendung, neu:
    import kopf
    kopf.ueberschrift(fig, "Temperaturmittel", "Global | Jahr | ERA5", "Abweichung von ...")
    fig.subplots_adjust(top=kopf.DIAGRAMM_OBEN, ...)

oder für bestehende Skripte (Titel per set_title + zwei annotate gebaut):
    kopf.setzen(fig, ax)       # übernimmt die drei Texte und setzt sie an die festen Stellen
"""

from matplotlib.text import Annotation

SCHRIFTART = "DejaVu Sans"

# Schriftgröße als Anteil der Bildhöhe (in Punkt)
GROESSE = (0.0486, 0.0309, 0.0251)
# Oberkante der Zeilen als Anteil der Bildhöhe von oben
OBEN = (0.0105, 0.069, 0.113)
# Oberkante des Diagramms (für subplots_adjust(top=...)); darüber ist Platz für die Überschrift
DIAGRAMM_OBEN = 0.85


def ueberschrift(fig, titel, mitte, zeile3, farbe=None):
    """Drei Zeilen an den festen Stellen; gibt die drei Textobjekte zurück."""
    hoehe_pt = fig.get_figheight() * 72
    stile = ("bold", "bold", "normal")
    texte = []
    for text, y, g, stil in zip((titel, mitte, zeile3), OBEN, GROESSE, stile):
        if not text:
            continue
        t = fig.text(0.5, 1 - y, text, ha="center", va="top", fontsize=g * hoehe_pt, fontweight=stil,
                     family=SCHRIFTART, color=farbe)
        texte.append(t)
    return texte


def setzen(fig, ax, oben=DIAGRAMM_OBEN):
    """Bestehende Überschrift (ax.set_title + annotate an ax.title + annotate darüber) an die
    festen Stellen setzen und das Diagramm oben passend begrenzen."""
    zeile3 = ax.get_title()
    mitte = next((a for a in ax.texts if isinstance(a, Annotation) and a.xycoords is ax.title), None)
    titel = next((a for a in ax.texts if isinstance(a, Annotation) and mitte is not None and a.xycoords is mitte), None)
    farbe = ax.title.get_color()
    texte = [titel.get_text() if titel else "", mitte.get_text() if mitte else "", zeile3]
    for a in (titel, mitte):
        if a is not None:
            a.remove()
    ax.set_title("")
    ueberschrift(fig, *texte, farbe=farbe)
    fig.subplots_adjust(top=min(fig.subplotpars.top, oben))
