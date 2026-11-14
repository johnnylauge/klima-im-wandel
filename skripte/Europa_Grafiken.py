"""
Europa (Landflächen 25° W–40° E, 34°–72° N) nach ERA5: Balkendiagramm und Liniendiagramm der
Jahresmittel, Abweichung von der vorindustriellen Zeit (1850-1900), gerechnet wie Copernicus.

Daten:
  1. 1940 bis 2025: ERA5-Spalte des Copernicus-Klimaindikators Europa
     (time_series_of_annual_temperatures_1850-2025_wrt_1850-1900_europe.csv, im Ordner startdaten).
     Copernicus setzt 1850-1900 um 1,43 K kälter an als 1991-2020 (steht im Kopf der CSV).
  2. Ab 2026: eigene Tagesreihe Europa_Land_Tag.txt (Europa_Download.py, täglich um 06 UTC auf
     GitHub). Laufendes Jahr = Mittel der täglichen Abweichungen vom Tages-Klimamittel 1991-2020
     (ohne 29. Februar, wie Copernicus) plus 1,43 K. Das Tages-Klimamittel wird aus den zwölf
     ERA5-Monatsmitteln 1991-2020 (Europa_Land_Klima_9120.txt) mittelwerttreu interpoliert.
  3. °C = Abweichung − 1,43 K + Klimamittel 1991-2020 (≈ 9,25 °C).
Sobald die vollständige Tagesreihe vom PC da ist (startdaten/Europa_Land_Tag.txt), wird sie mitgelesen.

Läuft lokal (Windows) und auf GitHub:
  ERA5_FOLDER      hierhin kommen TXT und Grafiken (auf GitHub _site/daten)
  EUROPA_DATEN     Ordner der Tagesreihe und des Klimamittels (auf GitHub europa_daten)
  EUROPA_CSV       Pfad zur Copernicus-CSV (sonst wird in ERA5_FOLDER, EUROPA_DATEN und startdaten gesucht)
  ERA5_DUNKEL=1    dunkle Fassung (..._dunkel.png)
"""

import os
import re
import json
import numpy as np

def tag_monat(d):
    """Datum kurz wie auf der Webseite, z. B. '31. Aug.' (Mai ohne Punkt)."""
    return f"{d.day}. " + ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.",
                           "Sep.", "Okt.", "Nov.", "Dez."][d.month - 1]

import pandas as pd
import angleichung      # gemeinsamer Bezug: 1850-1900, angeglichen über 1991-2020
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


# Achsenbeschriftung in deutscher Schreibweise (1,5 statt 1.5) - gilt für alle Achsen ohne eigene Formatierung
from matplotlib.ticker import ScalarFormatter as _SF
import kopf             # einheitliche Überschrift
if not getattr(_SF, "_komma", False):
    _sf_alt = _SF.__call__
    _SF.__call__ = lambda self, x, pos=None: _sf_alt(self, x, pos).replace(".", ",")
    _SF._komma = True


def dz(v, fmt=".2f"):
    """Zahl in deutscher Schreibweise für Grafiktexte: Komma statt Punkt, echtes Minus."""
    return format(v, fmt).replace(".", ",").replace("-", "−")

# =========================================================
# EINSTELLUNGEN
# =========================================================
WAHL = "Jahr"                   # vorerst nur das Jahr (Jahreszeiten und Monate folgen mit der vollen Tagesreihe)
ARTEN = ["balken", "linien"]    # welche Diagramme: "balken" und "linien" (beide Abweichung)
SHOW_TREND = True               # LOESS-Trend (30 Jahre)
SHOW_1_5_LINE = False           # 1,5-K-Linie (gilt fürs globale Mittel - bei Europa aus)
SHOW_RECORD = True              # wärmsten (vollständigen) Zeitraum markieren
SHOW_INFO = True                # Info-Kasten oben links
SHOW_UNSICHERHEIT = False       # keine Unsicherheit verfügbar
LINIE = 3.0                     # Stärke der farbigen Linie
X_TICK_STEP = 20                # Abstand der Jahreszahlen auf der x-Achse

PNG_DPI = 150                   # 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)
SCHRIFT = 1.3

DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
INFO_ALS_PLATZHALTER = SHOW_INFO and WEB
AUF_GITHUB = bool(os.environ.get("CI"))
ANZEIGEN = False                # wird unten gesetzt: nur bei einer Auswahl lokal ein Fenster
if os.environ.get("EUROPA_ARTEN"):
    ARTEN = [a.strip() for a in os.environ["EUROPA_ARTEN"].split(",")]

FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                fuss="gray", linie_rand="white", strich="dimgray", limit="#b2182b", linie="#6b6b70",
                band="#9a9aa2"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                fuss="#9a9aa2", linie_rand="#1e1e21", strich="#9a9aa2", limit="#ff6b6b", linie="#9a9aa2",
                band="#8a8a92"),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"], "hatch.color": FARBEN["fg"],
})

HIER = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Copernicus\Europa")
DATEN = os.environ.get("EUROPA_DATEN", FOLDER)
STARTDATEN = os.path.join(HIER, "..", "startdaten")
CSV_NAME = "time_series_of_annual_temperatures_1850-2025_wrt_1850-1900_europe.csv"
KLIMA_ERSATZ = 9.245              # Klimamittel 1991-2020 (°C), falls Europa_Land_Klima_9120.txt noch fehlt
OFFSET_ERSATZ = 1.43              # 1991-2020 über 1850-1900, falls nicht im Kopf der CSV
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")
QUELLE = ("Datenquelle: Copernicus Climate Change Service (C3S) · ERA5 · Klimaindikator Europa (bis 2025)\n"
          "Landfläche 25°W–40°E, 34°–72°N · ab 2026 eigene Berechnung aus ERA5-Tagesmitteln\n"
          "Visualisierung: JohnnyLauge.bsky.social")
MONTH_NAMES = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
               "August", "September", "Oktober", "November", "Dezember"]
MONATSTAGE = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]       # 365-Tage-Jahr


def komma(x, stellen=2):
    return f"{x:.{stellen}f}".replace(".", ",")


def k_text(x):
    return ("+" if x >= 0 else "−") + komma(abs(x)) + " K"


def loess(reihe, fenster=30):
    """LOESS-Glättung: lokale Gerade durch die 'fenster' nächstgelegenen Jahre (Tricube-Gewichte)."""
    x = reihe.index.to_numpy(float)
    y = reihe.to_numpy(float)
    q = min(fenster, len(x))
    werte = []
    for x0 in x:
        d = np.abs(x - x0)
        h = np.sort(d)[q - 1] * 1.000001 or 1.0
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        A = np.vstack([np.ones_like(x), x - x0]).T * np.sqrt(w)[:, None]
        werte.append(np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0][0])
    return pd.Series(werte, index=reihe.index)


def weich(x, y):
    """Weiche Kurve (Catmull-Rom-Spline) durch die Jahreswerte."""
    if len(x) < 3:
        return x, y
    px = np.concatenate([[2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]])
    py = np.concatenate([[2 * y[0] - y[1]], y, [2 * y[-1] - y[-2]]])
    t = np.linspace(0, 1, 20, endpoint=False)[:, None]
    xs, ys = [], []
    for p in ((px, xs), (py, ys)):
        p0, p1, p2, p3 = p[0][:-3], p[0][1:-2], p[0][2:-1], p[0][3:]
        seg = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                     + (3 * p1 - p0 - 3 * p2 + p3) * t**3)
        p[1].extend(seg.T.ravel())
    return np.append(xs, x[-1]), np.append(ys, y[-1])


# =========================================================
# DATEN EINLESEN
# =========================================================
os.makedirs(FOLDER, exist_ok=True)


def finde(name, *ordner):
    for o in ordner:
        p = os.path.join(o, name)
        if os.path.exists(p):
            return p
    return None


def lies_csv(datei):
    """ERA5-Spalte des Klimaindikators (K über 1850-1900) und der Abstand 1991-2020 -> 1850-1900."""
    text = open(datei, encoding="utf-8-sig").read()
    m = re.search(r"set at\s+([\d.]+)\s*C?\s+colder than 1991-2020", text)
    offset = float(m.group(1)) if m else OFFSET_ERSATZ
    zeilen = [z for z in text.splitlines() if z.strip() and not z.startswith("#")]
    kopfzeile = [s.strip() for s in zeilen[0].split(",")]
    sp = kopfzeile.index("ERA5")
    werte = {}
    for z in zeilen[1:]:
        t = z.split(",")
        try:
            jahr, v = int(t[0]), float(t[sp])
        except (ValueError, IndexError):
            continue
        if v > -990:
            werte[jahr] = v
    return pd.Series(werte, dtype=float).sort_index(), offset


def lies_klima():
    datei = finde("Europa_Land_Klima_9120.txt", DATEN, FOLDER, STARTDATEN)
    if not datei:
        return None
    k = pd.read_csv(datei, sep=";", comment="#", index_col="monat")["tmittel"]
    return k.reindex(range(1, 13)).to_numpy(float) if len(k) == 12 else None


def lies_tage():
    """Tagesreihen (GitHub-Download und ggf. die vom PC) zusammenführen: erst Duplikate weg, dann sortieren."""
    teile = []
    for o in dict.fromkeys([STARTDATEN, FOLDER, DATEN]):    # spätere Ordner haben Vorrang
        p = os.path.join(o, "Europa_Land_Tag.txt")
        if os.path.exists(p):
            teile.append(pd.read_csv(p, sep=";", parse_dates=["datum"], index_col="datum")["tmittel"])
    if not teile:
        return pd.Series(dtype=float)
    s = pd.concat(teile)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()


def tagesklima(monate):
    """365 Werte: aus den Monatsmitteln linear interpoliert und so nachgebessert, dass jedes
    Monatsmittel genau erhalten bleibt (mittelwerttreu)."""
    grenzen = np.concatenate([[0], np.cumsum(MONATSTAGE)])
    mitte = (grenzen[:-1] + grenzen[1:]) / 2
    tage = np.arange(365) + 0.5
    xp = np.concatenate([mitte - 365, mitte, mitte + 365])
    ziel = np.array(monate, float)
    for _ in range(100):
        kurve = np.interp(tage, xp, np.tile(ziel, 3))
        ist = np.array([kurve[grenzen[i]:grenzen[i + 1]].mean() for i in range(12)])
        ziel += np.array(monate) - ist
        if np.abs(np.array(monate) - ist).max() < 1e-6:
            break
    return kurve


def tag_365(index):
    """Tag im 365-Tage-Jahr (0-364); der 29. Februar bekommt -1 und wird weggelassen."""
    d = pd.DatetimeIndex(index)
    n = d.dayofyear.to_numpy() - 1
    schalt = d.is_leap_year
    n = np.where(schalt & (n >= 60), n - 1, n)
    return np.where(schalt & (d.month == 2) & (d.day == 29), -1, n)


CSV_DATEI = os.environ.get("EUROPA_CSV") or finde(CSV_NAME, FOLDER, DATEN, STARTDATEN)
if not CSV_DATEI:
    raise SystemExit(f"Copernicus-CSV {CSV_NAME} nicht gefunden (Ordner startdaten)")
CSV_WERTE, OFFSET = lies_csv(CSV_DATEI)
KLIMA_MONATE = lies_klima()
KLIMA_JAHR = (float(np.average(KLIMA_MONATE, weights=MONATSTAGE)) if KLIMA_MONATE is not None
              else KLIMA_ERSATZ)
print(f"Copernicus-Klimaindikator: ERA5 {CSV_WERTE.index.min()}-{CSV_WERTE.index.max()}, "
      f"1850-1900 = 1991-2020 − {OFFSET:.2f} K")
print(f"Klimamittel 1991-2020: {KLIMA_JAHR:.3f} °C"
      + ("" if KLIMA_MONATE is not None else " (Ersatzwert - Europa_Land_Klima_9120.txt fehlt noch)"))

zeilen = [dict(year=j, anomaly=v, temp=v - OFFSET + KLIMA_JAHR, days=366 if pd.Timestamp(j, 1, 1).is_leap_year else 365,
               last_date=pd.Timestamp(j, 12, 31), voll=True, quelle="csv") for j, v in CSV_WERTE.items()]
TAGE = lies_tage()
if len(TAGE) and KLIMA_MONATE is not None:
    klima_tag = tagesklima(KLIMA_MONATE)
    n = tag_365(TAGE.index)
    ab = pd.Series(TAGE.to_numpy() - np.where(n >= 0, klima_tag[np.clip(n, 0, 364)], np.nan), index=TAGE.index)
    ab = ab[n >= 0]
    # Prüfung: eigene Jahre, die auch im Klimaindikator stehen
    vergleich = []
    for j, a in ab.groupby(ab.index.year):
        if len(a) >= 365 and j in CSV_WERTE.index:
            vergleich.append(a.mean() + OFFSET - CSV_WERTE[j])
    if vergleich:
        print(f"Abgleich eigene Tagesreihe mit dem Klimaindikator ({len(vergleich)} Jahre): "
              f"größter Unterschied {max(vergleich, key=abs):+.3f} K")
    for j, a in ab.groupby(ab.index.year):
        if j in CSV_WERTE.index:
            continue                       # bis Copernicus das Jahr veröffentlicht: eigene Berechnung
        if not (a.index.min() <= pd.Timestamp(j, 1, 7)):
            print(f"{j}: Tagesreihe beginnt erst am {a.index.min():%d.%m.} - Jahr ausgelassen")
            continue
        zeilen.append(dict(year=j, anomaly=a.mean() + OFFSET, temp=a.mean() + KLIMA_JAHR,
                           days=int((TAGE.index.year == j).sum()), last_date=a.index.max(),
                           voll=a.index.max() >= pd.Timestamp(j, 12, 31) and len(a) >= 365, quelle="tage"))
elif len(TAGE):
    print("Tagesreihe vorhanden, aber noch kein Klimamittel 1991-2020 - laufendes Jahr folgt später")
else:
    print("Noch keine Tagesreihe ab 2026 - Grafik bis zum letzten Jahr des Klimaindikators")

JAHRE = pd.DataFrame(zeilen).set_index("year").sort_index()
JAHRE["unsicher"] = 0.0
LETZTER = JAHRE.iloc[-1]
STAND = f"{LETZTER['last_date']:%d.%m.%Y}"


def berechne():
    werte = JAHRE.copy()
    current_year = int(werte.index.max())
    is_partial = not bool(werte.loc[current_year, "voll"])
    werte["rank"] = werte["anomaly"].rank(ascending=False).astype(int)
    return werte, current_year, is_partial


def speichere_txt(werte, datei):
    with open(datei, "w", encoding="utf-8") as f:
        f.write("# Temperaturmittel Jahr – Europa (Land, 25°W–40°E, 34°–72°N; ERA5)\n"
                f"# Abweichung bezogen auf 1850-1900 (= 1991-2020 minus {komma(OFFSET)} K, wie Copernicus)\n"
                f"# Stand: {STAND}\n"
                f"# Quelle: Copernicus-Klimaindikator (ERA5) bis {CSV_WERTE.index.max()}, danach eigene Berechnung aus ERA5-Tagesmitteln\n"
                f"# °C = Klimamittel 1991-2020 ({komma(KLIMA_JAHR)} °C) + Abweichung − {komma(OFFSET)} K\n"
                "#\n"
                f"{'Jahr':>4}  {'Temp_°C':>8}  {'Abw_K':>6}  {'Rang':>4}  {'Tage':>4}  {'letzter_Tag':>11}\n")
        for jahr, z in werte.iterrows():
            f.write(f"{jahr:>4}  {z['temp']:>8.3f}  {z['anomaly']:>+6.3f}  {z['rank']:>4d}  "
                    f"{int(z['days']):>4d}  {z['last_date']:%d.%m.%Y}\n")
    print(f"Werte gespeichert unter:\n{datei}")


# =========================================================
# GEMEINSAME BAUSTEINE DER GRAFIKEN
# =========================================================
def grundgeruest(titel_mitte, zeile3):
    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    ax.set_title(zeile3, fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mitte = ax.annotate(titel_mitte, xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=24 * SCHRIFT, fontweight="bold")
    kopf.setzen(fig, ax)   # Überschrift an festen Stellen (siehe kopf.py)
    fig.text(0.99, 0.01, QUELLE, ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=kopf.DIAGRAMM_OBEN)
    return fig, ax


def achsen_beschriften(ax, ylabel):
    # keine Beschriftung der x-Achse: Jahreszahlen bzw. Monate erklären sich selbst
    ax.set_ylabel(ylabel, fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def info_kasten(ax, werte, current_year, is_partial, label):
    cur = werte.loc[current_year]
    bis = f" (bis {tag_monat(cur['last_date'])})" if is_partial else ""
    info = (f"Temperaturmittel {label(current_year)}{bis}: {dz(cur['temp'], '.2f')} °C\n"
            f"Temperaturabweichung 1850-1900: {dz(cur['anomaly'], '+.2f')} K  "
            f"({'bisher ' if is_partial else ''}Rang {cur['rank']})")
    t = ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                fontsize=10 * SCHRIFT, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
                          edgecolor=FARBEN["box_rand"], alpha=0.9))
    t.set_visible(SHOW_INFO)
    return t


def layout_und_kaesten(fig, ax, legend, info_text, xy_hindernisse, grenze, rekord):
    """Legende unter den Info-Kasten, y-Achse notfalls erweitern, Rekord-Kasten frei platzieren.
    xy_hindernisse: Liste von Arrays (Datenkoordinaten), die nichts verdecken darf.
    rekord: (Text, x, y) oder None."""

    def ausrichten():
        for _ in range(2):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            area = ax.get_window_extent(r)
            l = legend.get_frame().get_window_extent(r)
            if SHOW_INFO:
                k = info_text.get_bbox_patch().get_window_extent(r)
                gap = area.y1 - k.y1
                ziel_oben, links = k.y0 - gap, k.x0
            else:
                gap = 0.03 * area.height
                ziel_oben, links = area.y1 - gap, area.x0 + 0.01 * area.width
            if SHOW_1_5_LINE and grenze is not None:
                linie = ax.transData.transform((0, grenze))[1]
                if ziel_oben - l.height < linie + gap and ziel_oben > linie - gap:
                    ziel_oben = linie - gap
            anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
            legend.set_bbox_to_anchor((anker.x0 + (links - l.x0) / area.width,
                                       anker.y0 + (ziel_oben - l.y1) / area.height),
                                      transform=ax.transAxes)

    def verdeckt(bb, linie_mit=True):
        for xy in xy_hindernisse:
            p = ax.transData.transform(xy)
            if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                return True
        if linie_mit and SHOW_1_5_LINE and grenze is not None:
            y15 = ax.transData.transform((0, grenze))[1]
            if bb.y0 < y15 < bb.y1:
                return True
        return False

    def kopf():
        r = fig.canvas.get_renderer()
        rahmen = [legend.get_frame().get_window_extent(r).padded(4)]
        if SHOW_INFO:
            rahmen.append(info_text.get_bbox_patch().get_window_extent(r).padded(4))
        return rahmen

    for _ in range(12):
        ausrichten()
        if not any(verdeckt(bb, False) for bb in kopf()):
            break
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.05 * (hi - lo))

    if rekord and SHOW_RECORD:
        text, rx, ry = rekord

        def kasten(links, dy, dx):
            return ax.annotate(text, xy=(rx, ry), xytext=(-dx if links else dx, dy),
                               textcoords="offset points", ha="right" if links else "left", va="center",
                               fontsize=10 * SCHRIFT, fontweight="bold", linespacing=1.3, gid="rekord",
                               bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"]),
                               arrowprops=dict(arrowstyle="-", color=FARBEN["strich"], lw=1,
                                               connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                               relpos=(1, 0.5) if links else (0, 0.5),
                                               shrinkA=0, shrinkB=6),
                               zorder=9)

        gesetzt = None
        for _ in range(15):
            ausrichten()
            r = fig.canvas.get_renderer()
            belegt = kopf()
            area = ax.get_window_extent(r)
            for dy in range(24, 320, 8):
                for dx in (40, 80, 130, 200):
                    for links in (True, False):
                        m = kasten(links, dy, dx)
                        m.update_positions(r)
                        m.update_bbox_position_size(r)
                        bb = m.get_bbox_patch().get_window_extent(r).padded(4)
                        if (bb.x0 >= area.x0 and bb.x1 <= area.x1 and bb.y1 <= area.y1
                                and not any(bb.overlaps(o) for o in belegt) and not verdeckt(bb)):
                            gesetzt = m
                            break
                        m.remove()
                    if gesetzt:
                        break
                if gesetzt:
                    break
            if gesetzt:
                break
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.05 * (hi - lo))
        if gesetzt is None:
            kasten(True, 24, 40)
        ausrichten()


def speichern(fig, ax, info_text, name_basis, felder):
    """Volle Fassung (mit Kästen), interaktive Fassung und Daten für die Maus-Anzeige."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    info_rahmen = info_text.get_bbox_patch().get_window_extent(r) if SHOW_INFO else None
    os.makedirs(PNG_FOLDER, exist_ok=True)
    png_file = os.path.join(PNG_FOLDER, name_basis + ("_dunkel" if DUNKEL else "") + ".png")
    if INFO_ALS_PLATZHALTER:
        voll_file = os.path.splitext(png_file)[0] + "_voll.png"
        fig.savefig(voll_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
        fig.savefig(os.path.splitext(voll_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])   # PDF zum Herunterladen
        print(f"Grafik (mit Kästen) gespeichert unter:\n{voll_file}")
        info_text.set_visible(False)
        for t in ax.texts + ax.collections:
            if t.get_gid() == "rekord":
                t.set_visible(False)
    fig.savefig(png_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
    if not INFO_ALS_PLATZHALTER:                  # lokal: PDF neben dem PNG
        fig.savefig(os.path.splitext(png_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    print(f"Grafik gespeichert unter:\n{png_file}")
    if WEB:
        fig.canvas.draw()
        breite, hoehe = fig.canvas.get_width_height()
        fx = lambda v: round(min(max(v / breite, 0), 1), 5)
        fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
        achse = ax.get_window_extent(r)
        balken = []
        for f in felder():
            balken.append({**{k: v for k, v in f.items() if k not in ("px",)},
                           "x0": fx(f["px"][0]), "x1": fx(f["px"][1]),
                           "y0": fy(f["px"][3]), "y1": fy(f["px"][2])})
        with open(os.path.join(PNG_FOLDER, name_basis + ".json"), "w", encoding="utf-8") as fh:
            json.dump(dict(achse=dict(x0=fx(achse.x0), x1=fx(achse.x1), y0=fy(achse.y1), y1=fy(achse.y0)),
                           info=(dict(x0=fx(info_rahmen.x0), x1=fx(info_rahmen.x1),
                                      y0=fy(info_rahmen.y1), y1=fy(info_rahmen.y0))
                                 if info_rahmen is not None else None),
                           balken=balken), fh, ensure_ascii=False)
    if not ANZEIGEN:
        plt.close(fig)


# =========================================================
# BALKENDIAGRAMM (Abweichung von 1850-1900)
# =========================================================
def balken(werte, current_year, is_partial, art, name, label, teil):
    wort = "Jahr" if art == "jahr" else name
    fig, ax = grundgeruest(f"Europa (Land) | {wort} | ERA5", "Abweichung von der vorindustriellen Zeit (1850-1900)")
    lim = werte["anomaly"].abs().max()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")
    farben = cmap(norm(werte["anomaly"]))
    bars = ax.bar(werte.index, werte["anomaly"], width=0.8, color=farben,
                  edgecolor="gray", linewidth=0.3, zorder=3)
    if is_partial:
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor(FARBEN["fg"])
        bars[-1].set_linewidth(1.0)
    voll = werte.drop(current_year) if is_partial else werte
    if SHOW_TREND:
        t = loess(voll["anomaly"], 30)
        x, y = weich(t.index.to_numpy(float), t.to_numpy(float))
        ax.plot(x, y, color=FARBEN["fg"], lw=2.2, ls="--",
                path_effects=[pe.Stroke(linewidth=4, foreground=FARBEN["linie_rand"]), pe.Normal()],
                label="LOESS-Trend (30 Jahre)", zorder=5)
    if SHOW_1_5_LINE:
        ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=1.8, label="1,5-K-Grenze (Pariser Klimaabkommen)", zorder=2)
    ax.axhline(0, color=FARBEN["fg"], lw=0.8, zorder=2)
    info_text = info_kasten(ax, werte, current_year, is_partial, label)

    ax.set_xlim(werte.index.min() - 1, werte.index.max() + 1)
    lo = min(werte["anomaly"].min(), 0) - 0.1
    hi = max(werte["anomaly"].max(), 1.5) + 0.15
    ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xticks(range((werte.index.min() // X_TICK_STEP) * X_TICK_STEP + X_TICK_STEP
                        if werte.index.min() % X_TICK_STEP else werte.index.min(), current_year + 1, X_TICK_STEP))
    achsen_beschriften(ax, "Temperaturabweichung [K]")
    ax.grid(True, axis="y", alpha=0.2, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    if is_partial:
        handles.append(Patch(facecolor=farben[-1], edgecolor=FARBEN["fg"], hatch="///", linewidth=1.0))
        labels.append(f"{label(current_year)} unvollständig (bis {tag_monat(werte.loc[current_year, 'last_date'])})")
    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.80), frameon=True,
                       fontsize=10 * SCHRIFT, fancybox=True, framealpha=0.85, edgecolor=FARBEN["box_rand"],
                       facecolor=FARBEN["box_bg"], borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")
    legend.set_zorder(10)

    # Hindernisse: Balkenspitzen (fein abgetastet) und die Trendlinie
    spitzen = np.concatenate([np.column_stack([werte.index + d, np.maximum(werte["anomaly"], 0)])
                              for d in (-0.4, 0, 0.4)] +
                             [np.column_stack([werte.index + d, np.minimum(werte["anomaly"], 0)])
                              for d in (-0.4, 0.4)])
    saeulen = [np.column_stack([np.full(6, j), np.linspace(0, a, 6)]) for j, a in werte["anomaly"].items()]
    hindernisse = [spitzen, np.concatenate(saeulen)] + [l.get_xydata() for l in ax.get_lines()
                                                         if l.get_label().startswith("LOESS")]
    rek = voll["anomaly"].idxmax()
    w = "Wärmstes" if art == "jahr" else "Wärmster"
    rekord = (f"{w} {'Jahr ' if art == 'jahr' else ''}{label(rek)}\n"
              f"{komma(voll.loc[rek, 'temp'])} °C | {k_text(voll.loc[rek, 'anomaly'])}", rek, voll.loc[rek, "anomaly"])
    layout_und_kaesten(fig, ax, legend, info_text, hindernisse, 1.5, rekord)

    def felder():
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for b, (jahr, z) in zip(bars, werte.iterrows()):
            e = b.get_window_extent(r)
            yield dict(jahr=int(jahr), label=label(jahr), rang=int(z["rank"]), temp=round(float(z["temp"]), 3),
                       abw=round(float(z["anomaly"]), 3), laufend=bool(is_partial and jahr == current_year),
                       px=(e.x0, e.x1, min(e.y0, e.y1), max(e.y0, e.y1)))
    speichern(fig, ax, info_text, f"ERA5_europa_{teil}", felder)


# =========================================================
# LINIENDIAGRAMM (Abweichung von 1850-1900 in K)
# =========================================================
def linien(werte, current_year, is_partial, art, name, label, teil):
    wort = "Jahr" if art == "jahr" else name
    fig, ax = grundgeruest(f"Europa (Land) | {wort} | ERA5", "Abweichung von der vorindustriellen Zeit (1850-1900)")
    voll = werte.drop(current_year) if is_partial else werte
    abstand = float((voll["temp"] - voll["anomaly"]).median())
    werte = werte.copy()
    werte["plot"] = werte["anomaly"]                  # gezeichnet wird die Abweichung (K)
    werte["c"] = werte["temp"]                        # °C für Rekord-Kasten und Maus-Anzeige
    if is_partial:                             # laufender Zeitraum: °C über die Abweichung umgerechnet
        werte.loc[current_year, "c"] = werte.loc[current_year, "anomaly"] + abstand
    voll = werte.drop(current_year) if is_partial else werte
    lim = werte["anomaly"].abs().max()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    if SHOW_UNSICHERHEIT:
        ax.fill_between(werte.index, werte["plot"] - werte["unsicher"], werte["plot"] + werte["unsicher"],
                        color=FARBEN["band"], alpha=0.22, linewidth=0, zorder=1, label="Unsicherheit (95 %)")

    def farbige_linie(daten, gestrichelt=False):
        x = daten.index.to_numpy(float)
        y = daten["plot"].to_numpy(float)
        a = daten["anomaly"].to_numpy(float)
        if len(x) < 2:
            return
        t = np.linspace(0, 1, 9)
        xs = np.concatenate([x[i] + (x[i + 1] - x[i]) * t[:-1] for i in range(len(x) - 1)] + [x[-1:]])
        ys, an = np.interp(xs, x, y), np.interp(xs, x, a)
        pts = np.column_stack([xs, ys])
        stil = (0, (1.2, 1.4)) if gestrichelt else "solid"
        ax.plot(xs, ys, color=FARBEN["linie"], lw=LINIE + 1.4, ls=stil, zorder=3,
                solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round")
        ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1),
                                         colors=cmap(norm((an[:-1] + an[1:]) / 2)), linewidths=LINIE,
                                         linestyles=stil, capstyle="round", joinstyle="round", zorder=4))

    farbige_linie(voll)
    if is_partial and len(voll):
        farbige_linie(werte.loc[[voll.index[-1], current_year]], gestrichelt=True)

    if SHOW_TREND:
        t = loess(voll["plot"], 30)
        x, y = weich(t.index.to_numpy(float), t.to_numpy(float))
        ax.plot(x, y, color=FARBEN["fg"], lw=2.2, ls="--",
                path_effects=[pe.Stroke(linewidth=4, foreground=FARBEN["linie_rand"]), pe.Normal()],
                label="LOESS-Trend (30 Jahre)", zorder=5)
    grenze = 1.5
    if SHOW_1_5_LINE:
        ax.axhline(grenze, color=FARBEN["limit"], ls=":", lw=1.8, zorder=2,
                   label="1,5-K-Grenze (Pariser Klimaabkommen)")
    ax.axhline(0, color=FARBEN["fg"], lw=0.8, zorder=2)
    info_text = info_kasten(ax, werte, current_year, is_partial, label)

    ax.set_xlim(werte.index.min() - 1, werte.index.max() + 1)
    lo = (werte["plot"] - werte["unsicher"]).min() - 0.1
    hi = max((werte["plot"] + werte["unsicher"]).max(), grenze) + 0.15
    ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
    ax.yaxis.set_major_locator(MultipleLocator(0.25 if hi - lo < 3 else 0.5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: komma(v)))
    ax.set_xticks(range((werte.index.min() // X_TICK_STEP) * X_TICK_STEP + X_TICK_STEP
                        if werte.index.min() % X_TICK_STEP else werte.index.min(), current_year + 1, X_TICK_STEP))
    achsen_beschriften(ax, "Temperaturabweichung [K]")
    ax.grid(True, alpha=0.2, zorder=0)

    class Verlauf(HandlerBase):
        def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
            xs = np.linspace(-xd, -xd + w, 25)
            pts = np.column_stack([xs, np.full_like(xs, h / 2 - yd)])
            return [LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors=cmap(np.linspace(0.1, 0.95, 24)),
                                   linewidths=LINIE, capstyle="butt", transform=trans)]

    verlauf = object()
    handles, labels = ax.get_legend_handles_labels()
    mittel = {"jahr": "Jahresmittel", "saison": "Jahreszeitenmittel", "monat": "Monatsmittel"}[art]
    handles.insert(0, verlauf)
    labels.insert(0, f"{mittel} (Farbe nach Abweichung)")
    if is_partial:
        handles.append(Line2D([], [], color=cmap(norm(werte.loc[current_year, "anomaly"])), lw=LINIE,
                              ls=(0, (1.2, 1.4)), dash_capstyle="round"))
        labels.append(f"{label(current_year)} unvollständig "
                      f"(bis {tag_monat(werte.loc[current_year, 'last_date'])})")
    legend = ax.legend(handles, labels, handler_map={verlauf: Verlauf()}, loc="upper left",
                       bbox_to_anchor=(0.01, 0.80), frameon=True, fontsize=10 * SCHRIFT, fancybox=True,
                       framealpha=0.85, edgecolor=FARBEN["box_rand"], facecolor=FARBEN["box_bg"],
                       borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")
    legend.set_zorder(10)

    pkt = werte["plot"].reset_index().to_numpy(float)
    xs = np.linspace(pkt[0, 0], pkt[-1, 0], len(pkt) * 6)
    hindernisse = [pkt, np.column_stack([xs, np.interp(xs, pkt[:, 0], pkt[:, 1])])]
    hindernisse += [l.get_xydata() for l in ax.get_lines() if l.get_label().startswith("LOESS")]
    rek = voll["anomaly"].idxmax()
    ax.scatter([rek], [voll.loc[rek, "plot"]], s=150, facecolor="none", edgecolor=FARBEN["fg"],
               linewidth=1.6, zorder=6, gid="rekord")
    w = "Wärmstes" if art == "jahr" else "Wärmster"
    rekord = (f"{w} {'Jahr ' if art == 'jahr' else ''}{label(rek)}\n"
              f"{komma(voll.loc[rek, 'c'])} °C | {k_text(voll.loc[rek, 'anomaly'])}", rek, voll.loc[rek, "plot"])
    layout_und_kaesten(fig, ax, legend, info_text, hindernisse, grenze, rekord)

    def felder():
        for jahr, z in werte.iterrows():
            px, py = ax.transData.transform((jahr, z["plot"]))
            x0 = ax.transData.transform((jahr - 0.45, 0))[0]
            x1 = ax.transData.transform((jahr + 0.45, 0))[0]
            lauf = bool(is_partial and jahr == current_year)
            yield dict(jahr=int(jahr), label=label(jahr), rang=int(z["rank"]), temp=round(float(z["c"]), 3),
                       abw=round(float(z["anomaly"]), 3), laufend=lauf, ca=lauf,
                       px=(min(x0, px - 8), max(x1, px + 8), py - 9, py + 9))
    speichern(fig, ax, info_text, f"ERA5_europa_linien_{teil}", felder)


# =========================================================
# JAHR
# =========================================================
ANZEIGEN = not AUF_GITHUB


def label(jahr):
    return str(jahr)


werte, current_year, is_partial = berechne()
if not DUNKEL:                                  # TXT nur einmal (beim hellen Lauf) schreiben
    speichere_txt(werte, os.path.join(FOLDER, "ERA5_europa_jahreswerte.txt"))
print(f"{current_year}: {werte.loc[current_year, 'temp']:.2f} °C | "
      f"{werte.loc[current_year, 'anomaly']:+.2f} K | {'bisher ' if is_partial else ''}"
      f"Rang {werte.loc[current_year, 'rank']}")
if "balken" in ARTEN:
    balken(werte, current_year, is_partial, "jahr", "Jahr", label, "jahreswerte")
if "linien" in ARTEN:
    linien(werte, current_year, is_partial, "jahr", "Jahr", label, "jahr")
if ANZEIGEN:
    plt.show()
