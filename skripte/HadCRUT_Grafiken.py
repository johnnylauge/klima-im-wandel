"""
Globale Temperatur nach HadCRUT5 (Met Office Hadley Centre und Climatic Research Unit der
University of East Anglia, Land + Ozean, Monatswerte ab 1850): Balkendiagramme der Abweichung
von 1850-1900 und Liniendiagramme der Abweichung, jeweils für das Jahr,
die vier Jahreszeiten und die zwölf Monate.

Ablauf:
  1. Sucht auf der HadCRUT5-Seite die aktuelle Version (z. B. HadCRUT.5.2.0.0) und lädt die
     globale Monatsreihe (CSV, "analysis") herunter; gespeichert als HadCRUT5_global_monat.csv.
     Die Abweichungen beziehen sich dort auf 1961-1990, dazu gibt es die 95-%-Unsicherheit.
  2. Rechnet auf 1850-1900 um: Je Kalendermonat wird das Mittel 1850-1900 abgezogen.
  3. Bildet je Jahr das Mittel des Zeitraums (Jahr, Jahreszeit, Monat), mit Rang.
  4. HadCRUT5 liefert keine absoluten Werte. Für °C wird je Monat ein Mittel 1961-1990
     geschätzt: die Monatsmittel des 20. Jahrhunderts von NOAA, verschoben um das Mittel
     1901-2000 der HadCRUT5-Abweichungen. Im Jahresmittel ergibt das rund 14,0 °C für
     1961-1990 - passend zum Wert, den die HadCRUT-Gruppe selbst nennt.
  5. Speichert die Werte als TXT (gleicher Aufbau wie bei den anderen Quellen) und zeichnet
     die Grafiken im gleichen Stil wie die anderen.

Läuft lokal (Windows) und auf GitHub:
  - Auf GitHub gibt die Steuerdatei über Umgebungsvariablen vor:
      ERA5_FOLDER / HADCRUT_FOLDER  Datenordner im Projekt
      HADCRUT_WAHL                  z. B. "alle" (überschreibt WAHL)
      ERA5_DUNKEL=1                 dunkle Fassung (..._dunkel.png)
      HADCRUT_OHNE_DOWNLOAD=1       vorhandene Datei nutzen statt neu zu laden
"""

import os
import re
import json
import time
import requests
import numpy as np


def tag_monat(d):
    """Datum kurz wie auf der Webseite, z. B. '31. Aug.' (Mai ohne Punkt)."""
    return f"{d.day}. " + ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.",
                           "Sep.", "Okt.", "Nov.", "Dez."][d.month - 1]

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# =========================================================
# EINSTELLUNGEN
# =========================================================
WAHL = "alle"                   # "Jahr", eine Jahreszeit ("Herbst"), ein Monat ("Oktober" oder 10),
                                # eine Liste wie ["Jahr", "Sommer", 7] oder "alle" = alle 17 Zeiträume
ARTEN = ["balken", "linien"]    # welche Diagramme: "balken" (Abweichung), "linien" (°C, geschätzt)
SHOW_TREND = True               # LOESS-Trend (30 Jahre)
SHOW_1_5_LINE = True            # 1,5-K-Linie
SHOW_RECORD = True              # wärmsten (vollständigen) Zeitraum markieren
SHOW_INFO = True                # Info-Kasten oben links
SHOW_UNSICHERHEIT = True        # Unsicherheitsband (95 %) im Liniendiagramm
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
if os.environ.get("HADCRUT_WAHL"):
    WAHL = os.environ["HADCRUT_WAHL"]
if os.environ.get("HADCRUT_ARTEN"):
    ARTEN = [a.strip() for a in os.environ["HADCRUT_ARTEN"].split(",")]

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

# Die Versionsnummer steht in der Adresse; die aktuelle wird beim Download von der Übersichtsseite gelesen
UEBERSICHT = "https://www.metoffice.gov.uk/hadobs/hadcrut5/"
VERSION_ERSATZ = "HadCRUT.5.2.0.0"
URL_MUSTER = ("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/{v}/analysis/diagnostics/"
              "{v}.analysis.summary_series.global.monthly.csv")
FOLDER = os.environ.get("HADCRUT_FOLDER") or os.environ.get(
    "ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\HadCRUT5")
FILE = os.path.join(FOLDER, "HadCRUT5_global_monat.csv")
HERKUNFT = os.path.join(FOLDER, "HadCRUT5_herkunft.txt")   # Version und Abrufdatum (für den Lizenzhinweis)
# Monatsmittel des 20. Jahrhunderts (NOAA, Land + Ozean) - Grundlage für die geschätzten °C-Werte
ABSOLUT_20JH = [12.0, 12.1, 12.7, 13.7, 14.8, 15.5, 15.8, 15.6, 15.0, 14.0, 12.9, 12.2]
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")
QUELLE = ("Datenquelle: HadCRUT5 · © British Crown Copyright, Met Office · Open Government Licence v3.0\n"
          "https://www.metoffice.gov.uk/hadobs/hadcrut5/ · eigene Berechnung\n"
          "Visualisierung: JohnnyLauge.bsky.social")

SEASONS = {
    "Frühling": ([3, 4, 5], "fruehling"),
    "Sommer":   ([6, 7, 8], "sommer"),
    "Herbst":   ([9, 10, 11], "herbst"),
    "Winter":   ([12, 1, 2], "winter"),
}
MONTH_NAMES = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
               "August", "September", "Oktober", "November", "Dezember"]


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
# DOWNLOAD
# =========================================================
os.makedirs(FOLDER, exist_ok=True)
if os.environ.get("HADCRUT_OHNE_DOWNLOAD") == "1" and os.path.exists(FILE):
    print(f"Vorhandene Daten werden genutzt:\n{FILE}")
else:
    print("Lade aktuelle HadCRUT5-Daten herunter...")
    try:                                        # aktuelle Version suchen, z. B. HadCRUT.5.2.0.0
        seite = requests.get(UEBERSICHT, timeout=60).text
        versionen = set(re.findall(r"HadCRUT\.5\.\d+\.\d+\.\d+", seite))
        VERSION = max(versionen, key=lambda v: [int(x) for x in v.split(".")[1:]]) if versionen else VERSION_ERSATZ
    except requests.RequestException:
        VERSION = VERSION_ERSATZ
    print(f"Version: {VERSION}")
    URL = URL_MUSTER.format(v=VERSION)
    for versuch in range(1, 4):
        try:
            response = requests.get(URL, timeout=90)
            if response.status_code == 404 and URL != URL_MUSTER.format(v=VERSION_ERSATZ):
                URL = URL_MUSTER.format(v=VERSION_ERSATZ)     # neue Version noch ohne Datei
                continue
            response.raise_for_status()
            break
        except requests.RequestException as fehler:
            if versuch == 3:
                raise
            print(f"Download fehlgeschlagen ({fehler}) - neuer Versuch in 60 Sekunden ...")
            time.sleep(60)
    with open(FILE, "wb") as f:
        f.write(response.content)
    # Version (aus der tatsächlich genutzten Adresse) und Abrufdatum merken - die Lizenz verlangt beides
    genutzt = re.search(r"HadCRUT\.5\.\d+\.\d+\.\d+", URL).group(0)
    with open(HERKUNFT, "w", encoding="utf-8") as f:
        f.write(f"{genutzt}\n{pd.Timestamp.now():%d.%m.%Y}\n")
    print(f"Download abgeschlossen.\nDatei gespeichert unter:\n{FILE}")

# Version und Abrufdatum (auch wenn ohne neuen Download gerechnet wird)
try:
    VERSION_GENUTZT, ABGERUFEN = open(HERKUNFT, encoding="utf-8").read().split()[:2]
except (OSError, ValueError):
    VERSION_GENUTZT, ABGERUFEN = VERSION_ERSATZ, f"{pd.Timestamp.now():%d.%m.%Y}"


# =========================================================
# DATEI EINLESEN
# =========================================================
def lies_hadcrut(datei):
    """Liest die CSV: Time (JJJJ-MM), Anomaly, untere und obere Grenze (95 %)."""
    reihen = []
    for z in open(datei, encoding="utf-8", errors="replace").read().splitlines():
        teile = [t.strip() for t in z.split(",")]
        if len(teile) < 4 or not re.fullmatch(r"\d{4}-\d{2}", teile[0]):
            continue
        try:
            abw, unten, oben = float(teile[1]), float(teile[2]), float(teile[3])
        except ValueError:
            continue
        if not np.isfinite(abw):
            continue
        reihen.append((int(teile[0][:4]), int(teile[0][5:]), abw, (oben - unten) / 2))
    if not reihen:
        raise ValueError(f"Keine Monatswerte in {datei} gefunden - Aufbau der Datei prüfen")
    df = pd.DataFrame(reihen, columns=["jahr", "monat", "abw_6190", "unsicher"]).drop_duplicates(["jahr", "monat"])
    return df.sort_values(["jahr", "monat"]).reset_index(drop=True)


MONATE_DF = lies_hadcrut(FILE)
# Bezug 1850-1900 je Kalendermonat
bezug = MONATE_DF[(MONATE_DF["jahr"] >= 1850) & (MONATE_DF["jahr"] <= 1900)].groupby("monat")["abw_6190"].mean()
MONATE_DF["ano_pre"] = MONATE_DF["abw_6190"] - MONATE_DF["monat"].map(bezug)
# Geschätzte absolute Temperatur: Monatsmittel 1961-1990 = NOAA-Mittel 1901-2000 minus dem
# HadCRUT5-Mittel 1901-2000 (bezogen auf 1961-1990)
jh20 = MONATE_DF[(MONATE_DF["jahr"] >= 1901) & (MONATE_DF["jahr"] <= 2000)].groupby("monat")["abw_6190"].mean()
ABSOLUT_6190 = {m: ABSOLUT_20JH[m - 1] - jh20[m] for m in range(1, 13)}
MONATE_DF["temp"] = MONATE_DF["abw_6190"] + MONATE_DF["monat"].map(ABSOLUT_6190)
print(f"Geschätztes Mittel 1961-1990: {np.mean(list(ABSOLUT_6190.values())):.2f} °C")
MONATE_DF["tage"] = [pd.Period(f"{j}-{m:02d}").days_in_month for j, m in zip(MONATE_DF["jahr"], MONATE_DF["monat"])]
LETZTER = MONATE_DF.iloc[-1]
STAND = f"{MONTH_NAMES[int(LETZTER['monat']) - 1]} {int(LETZTER['jahr'])}"
print(f"HadCRUT5: {MONATE_DF['jahr'].min()}-{STAND}, {len(MONATE_DF)} Monate")



def zeitraum(wahl):
    """(Art, Name, Monate, Dateiteil) für TXT und Grafiken."""
    if str(wahl).lower() == "jahr":
        return "jahr", "Jahr", list(range(1, 13)), "jahreswerte", "jahr"
    for name, (monate, datei) in SEASONS.items():
        if str(wahl).lower() in (name.lower(), datei):
            return "saison", name, monate, f"saison_{datei}", f"saison_{datei}"
    if str(wahl).isdigit() and 1 <= int(wahl) <= 12:
        m = int(wahl)
    elif str(wahl).capitalize() in MONTH_NAMES:
        m = MONTH_NAMES.index(str(wahl).capitalize()) + 1
    else:
        raise ValueError(f"Unbekannte Auswahl '{wahl}'")
    return "monat", MONTH_NAMES[m - 1], [m], f"monat_{m:02d}", f"monat_{m:02d}"


def berechne(art, name, monate):
    """Mittel je Jahr des Zeitraums: temp (°C), anomaly (K zu 1850-1900), unsicher, Rang."""
    df = MONATE_DF[MONATE_DF["monat"].isin(monate)].copy()
    df["year"] = df["jahr"]
    if name == "Winter":
        df.loc[df["monat"] == 12, "year"] += 1
    werte = df.groupby("year").agg(temp=("temp", "mean"), anomaly=("ano_pre", "mean"),
                                   n=("monat", "size"), days=("tage", "sum"),
                                   unsicher=("unsicher", "mean"),   # vorsichtig: Fehler der Monate hängen zusammen, daher nicht kleiner gerechnet
                                   letzter_jahr=("jahr", "last"), letzter_monat=("monat", "last"))
    current_year = werte.index.max()
    voll = werte["n"] >= len(monate)
    erstes_volles = werte.index[voll].min()
    werte = werte[voll | (werte.index == current_year)].copy()
    werte = werte[werte.index >= erstes_volles]                     # unvollständigen Anfang weglassen
    werte["last_date"] = [pd.Timestamp(int(j), int(m), 1) + pd.offsets.MonthEnd(0)
                          for j, m in zip(werte["letzter_jahr"], werte["letzter_monat"])]
    is_partial = werte.loc[current_year, "n"] < len(monate)
    werte["rank"] = werte["anomaly"].rank(ascending=False).astype(int)
    return werte, current_year, is_partial


def speichere_txt(werte, art, name, datei):
    with open(datei, "w", encoding="utf-8") as f:
        f.write(f"# Temperaturmittel {name} – Global (HadCRUT5, Land + Ozean; °C geschätzt)\n"
                f"# Abweichung bezogen auf 1850-1900\n"
                f"# Stand: {STAND}\n"
                f"# Version: {VERSION_GENUTZT}\n"
                f"# Abgerufen: {ABGERUFEN}\n"
                + ("# Jahr = Jahr von Januar/Februar (2026 = Winter 2025/26)\n" if name == "Winter" else "")
                + "#\n"
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
    fig.text(0.99, 0.01, QUELLE, ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.87)
    return fig, ax


def achsen_beschriften(ax, ylabel):
    ax.set_xlabel("Jahr", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def info_kasten(ax, werte, current_year, is_partial, label):
    cur = werte.loc[current_year]
    bis = f" (bis {tag_monat(cur['last_date'])})" if is_partial else ""
    info = (f"Temperaturmittel {label(current_year)}{bis}: {cur['temp']:.2f} °C\n"
            f"Temperaturabweichung 1850-1900: {cur['anomaly']:+.2f} K  "
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
    fig, ax = grundgeruest(f"Global | {wort}", "Abweichung von der vorindustriellen Zeit (1850-1900)")
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
    speichern(fig, ax, info_text, f"HADCRUT_global_{teil}", felder)


# =========================================================
# LINIENDIAGRAMM (Abweichung von 1850-1900 in K)
# =========================================================
def linien(werte, current_year, is_partial, art, name, label, teil):
    wort = "Jahr" if art == "jahr" else name
    fig, ax = grundgeruest(f"Global | {wort}", "Abweichung von der vorindustriellen Zeit (1850-1900)")
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
    speichern(fig, ax, info_text, f"HADCRUT_global_linien_{teil}", felder)


# =========================================================
# AUSWAHL DURCHRECHNEN
# =========================================================
if str(WAHL).lower() == "alle":
    auswahl = ["Jahr"] + list(SEASONS) + list(range(1, 13))
elif isinstance(WAHL, (list, tuple)):
    auswahl = list(WAHL)
else:
    auswahl = [p.strip() for p in str(WAHL).split(",")]

ANZEIGEN = len(auswahl) == 1 and not AUF_GITHUB
for w in auswahl:
    art, name, monate, txt_teil, bild_teil = zeitraum(w)

    def label(jahr, art=art, name=name):
        if art == "jahr":
            return str(jahr)
        if name == "Winter":
            return f"Winter {jahr - 1}/{str(jahr)[-2:]}"
        return f"{name} {jahr}"

    werte, current_year, is_partial = berechne(art, name, monate)
    if not DUNKEL:                              # TXT nur einmal (beim hellen Lauf) schreiben
        speichere_txt(werte, art, name, os.path.join(FOLDER, f"HADCRUT_global_{txt_teil}.txt"))
    print(f"{label(current_year)}: {werte.loc[current_year, 'temp']:.2f} °C | "
          f"{werte.loc[current_year, 'anomaly']:+.2f} K | {'bisher ' if is_partial else ''}"
          f"Rang {werte.loc[current_year, 'rank']}")
    if "balken" in ARTEN:
        balken(werte, current_year, is_partial, art, name, label,
               "jahreswerte" if art == "jahr" else bild_teil)
    if "linien" in ARTEN:
        linien(werte, current_year, is_partial, art, name, label, bild_teil)
if ANZEIGEN:
    plt.show()
