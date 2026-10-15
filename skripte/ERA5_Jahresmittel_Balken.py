"""
Jahresmitteltemperatur global (ERA5) als Balkendiagramm der Abweichung, ab 1940.

Ablauf:
  1. Lädt die aktuellen Tageswerte der globalen 2-m-Temperatur (ERA5, Copernicus Climate
     Pulse) herunter und speichert sie als ERA5_global_tag.txt.
  2. Rechnet die Abweichung auf den vorindustriellen Bezug 1850-1900 um (Abweichung zu
     1991-2020 plus monatlicher Offset) - oder nutzt 1991-2020, je nach BASELINE.
  3. Bildet daraus Jahresmittel; das laufende Jahr wird als unvollständig markiert.
  4. Speichert die Werte mit Rang als TXT-Datei und gibt sie in der Konsole aus.
  5. Zeichnet ein Balkendiagramm (16:9 für Bluesky): Balken je Jahr in Blau-Rot nach
     Abweichung, gleitendes Mittel (RUNNING_WINDOW Jahre), 1,5-K-Linie, Info-Kasten oben links,
     Markierung des wärmsten vollständigen Jahres und Legende darunter.
  6. Speichert die Grafik als PNG im Unterordner "Grafiken".

Wichtige Schalter (Einstellungen unten):
  SHOW_CURRENT_YEAR   laufenden Zeitraum zeigen oder weglassen
  LAUFEND_FARBIG      laufenden Balken weiß (Standard) oder farbig
  SHOW_RECORD         wärmsten Zeitraum in der Grafik markieren
  BASELINE            Bezug 1850-1900 oder 1991-2020

Läuft lokal (Windows) und auf GitHub:
  - Lokal wird der Windows-Ordner unten bei FOLDER benutzt und die Grafik im Fenster gezeigt.
  - Auf GitHub gibt die Steuerdatei den Ordner über die Umgebungsvariable ERA5_FOLDER vor;
    dort öffnet sich kein Fenster (kein Bildschirm), die Grafik wird nur gespeichert.
  - DUNKEL = True (bzw. ERA5_DUNKEL=1) erzeugt die dunkle Fassung ..._dunkel.png.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
import json


def speichere_hover_daten(fig, ax, bars, werte, beschriftung, laufend_jahr, datei, info_rahmen=None):
    """Schreibt für die Webseite, wo jeder Balken im Bild liegt (Anteile 0..1 von Breite/Höhe,
    gemessen von oben links) und welche Werte dazugehören. Die Webseite legt damit unsichtbare
    Felder über das PNG und zeigt beim Darüberfahren mit der Maus einen Kasten an."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    breite, hoehe = fig.canvas.get_width_height()
    fx = lambda v: round(min(max(v / breite, 0), 1), 5)
    fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
    achse = ax.get_window_extent(r)
    balken = []
    for b, (jahr, z) in zip(bars, werte.iterrows()):
        e = b.get_window_extent(r)
        balken.append(dict(
            jahr=int(jahr), label=beschriftung(jahr), rang=int(z["rank"]),
            temp=round(float(z["temp"]), 3), abw=round(float(z["anomaly"]), 3),
            laufend=bool(jahr == laufend_jahr),
            x0=fx(e.x0), x1=fx(e.x1),
            y0=fy(max(e.y0, e.y1)), y1=fy(min(e.y0, e.y1)),   # oben / unten im Bild
        ))
    with open(datei, "w", encoding="utf-8") as f:
        json.dump(dict(achse=dict(x0=fx(achse.x0), x1=fx(achse.x1),
                                  y0=fy(achse.y1), y1=fy(achse.y0)),
                       info=(dict(x0=fx(info_rahmen.x0), x1=fx(info_rahmen.x1),
                                  y0=fy(info_rahmen.y1), y1=fy(info_rahmen.y0))
                             if info_rahmen is not None else None),
                       balken=balken), f, ensure_ascii=False)
    print(f"Daten für die Maus-Anzeige gespeichert unter:\n{datei}")


def loess(reihe, fenster=30):
    """LOESS-Glättung: für jedes Jahr eine lokale Gerade durch die 'fenster' nächstgelegenen
    Jahre, nahe Jahre zählen stärker (Tricube-Gewichte). Ergibt einen ruhigen Klimatrend,
    der - anders als ein zentriertes Mittel - bis zum letzten vollständigen Jahr reicht."""
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

# =========================================================
# EINSTELLUNGEN
# =========================================================
START_YEAR = 1940

BASELINE = "1850-1900"          # "1850-1900" (vorindustriell) oder "1991-2020"

SHOW_CURRENT_YEAR = True        # laufendes (unvollständiges) Jahr in der Grafik zeigen; False = nur volle Jahre
LAUFEND_FARBIG = True          # False = laufender Balken weiß (schraffiert); True = in der Farbe
                                # seines Wertes (schraffiert) - der Wert ist ein Mittel und damit
                                # schon früh mit den vollen Zeiträumen vergleichbar
SHOW_VALUES = True              # Abweichung als Zahl im Balken an der Nulllinie
SHOW_RUNNING_MEAN = True        # Gleitendes Mittel als Linie
RUNNING_WINDOW = 5              # Fensterbreite in Jahren
SMOOTH_LINE = True              # Linie weich statt eckig zeichnen
SHOW_RECORD = True              # wärmstes (vollständiges) Jahr in der Grafik markieren
SHOW_INFO = True                # Info-Kasten oben links (aktueller Wert und Rang)
TREND = "loess30"               # Trendlinie: "5jahre" = gleitendes 5-Jahres-Mittel,
                                # "loess30" = LOESS-Glättung über 30 Jahre (Klimatrend)
SHOW_1_5_LINE = True            # 1,5-K-Linie (nur bei Bezug 1850–1900)
Y_LIMITS = (-0.30, 1.80)        # feste y-Achse (unten, oben); None = automatisch
Y_TICK_STEP = 0.25              # Abstand der Beschriftung auf der y-Achse

SAVE_PNG = True                 # Grafik als PNG im Datenordner speichern
PNG_DPI = 150                   # Auflösung: 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)           # Bildgröße in Zoll; 16:9 passt für Bluesky (wie die DWD-Grafiken)
SCHRIFT = 1.3                   # Faktor für alle Schriftgrößen (1.0 = klein, 1.3 = wie DWD-Grafiken)

# Helle oder dunkle Grafik. Lokal hier umstellen; auf GitHub setzt die Steuerdatei
# ERA5_DUNKEL=1 und erzeugt so zusätzlich eine dunkle Fassung für die Webseite.
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"

# Webseiten-Fassung (auf GitHub automatisch): ohne Zahlen. Es entstehen zwei Bilder:
#   ..._voll.png  mit Info-Kasten und Rekord-Kasten (zum Vergrößern und Herunterladen)
#   ....png       ohne die beiden Kästen für die interaktive Ansicht - dort zeigt die Webseite
#                 die Werte des Balkens unter der Maus an der Stelle des Info-Kastens Lokal bleibt alles wie eingestellt.
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
INFO_ALS_PLATZHALTER = False     # zusätzlich ein Bild ohne Info- und Rekord-Kasten speichern
if WEB:
    SHOW_VALUES = False
    INFO_ALS_PLATZHALTER = SHOW_INFO  # dort zeigt die Webseite die Werte des Balkens unter der Maus
if os.environ.get("ERA5_TREND"):                 # Trendlinie auch von außen wählbar
    TREND = os.environ["ERA5_TREND"]

# Farben für helle und dunkle Grafik
FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                zahl="dimgray", fuss="gray", linie_rand="white", strich="dimgray", limit="#b2182b"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                zahl="#b9b9c0", fuss="#9a9aa2", linie_rand="#1e1e21", strich="#9a9aa2", limit="#ff6b6b"),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"], "hatch.color": FARBEN["fg"],
})

# Vorindustrielle Offsets je Monat (Januar bis Dezember)
MONTHLY_OFFSETS = dict(zip(range(1, 13), [
    0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93,
]))


URL = ("https://sites.ecmwf.int/data/climatepulse/data/series/"
       "era5_daily_series_2t_global.csv")
# Lokal: Windows-Ordner. Auf GitHub setzt die Steuerdatei ERA5_FOLDER auf einen Ordner im Projekt.
FOLDER = os.environ.get("ERA5_FOLDER",
                        r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
FILE = os.path.join(FOLDER, "ERA5_global_tag.txt")              # Tageswerte (Original)
ANNUAL_FILE = os.path.join(FOLDER, "ERA5_global_jahreswerte.txt")  # Jahreswerte
PNG_NAME = "ERA5_global_jahreswerte" + ("_dunkel" if DUNKEL else "") + ".png"

# =========================================================
# DOWNLOAD
# =========================================================
os.makedirs(FOLDER, exist_ok=True)
if os.environ.get("ERA5_OHNE_DOWNLOAD") == "1" and os.path.exists(FILE):
    print(f"Vorhandene Daten werden genutzt:\n{FILE}")
else:
    print("Lade aktuelle ERA5-Daten herunter...")
    for versuch in range(1, 4):                    # bis zu 3 Versuche, falls der Server hakt
        try:
            response = requests.get(URL, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException as fehler:
            if versuch == 3:
                raise
            print(f"Download fehlgeschlagen ({fehler}) - neuer Versuch in 60 Sekunden ...")
            time.sleep(60)
    with open(FILE, "wb") as f:
        f.write(response.content)
    print(f"Download abgeschlossen.\nDatei gespeichert unter:\n{FILE}")

# =========================================================
# DATEN AUFBEREITEN
# =========================================================
df = pd.read_csv(FILE, comment="#", parse_dates=["date"])
df = df[df["date"].dt.year >= START_YEAR]
df["year"] = df["date"].dt.year
df["ano_pre"] = df["ano_91-20"] + df["date"].dt.month.map(MONTHLY_OFFSETS)

ano_col = "ano_pre" if BASELINE == "1850-1900" else "ano_91-20"

annual = df.groupby("year").agg(
    temp=("2t", "mean"),
    anomaly=(ano_col, "mean"),
    days=("2t", "size"),
    last_date=("date", "max"),
)

current_year = annual.index.max()
last_date = annual.loc[current_year, "last_date"]
is_partial = last_date < pd.Timestamp(current_year, 12, 31)

# Rangliste (1 = wärmstes Jahr)
annual["rank"] = annual["anomaly"].rank(ascending=False).astype(int)

# =========================================================
# JAHRESWERTE ALS TXT SPEICHERN
# =========================================================
with open(ANNUAL_FILE, "w", encoding="utf-8") as f:
    f.write("# Jahresmitteltemperatur – Global (ERA5, Copernicus Climate Pulse)\n"
            f"# Abweichung bezogen auf {BASELINE}\n"
            f"# Stand: {last_date:%d.%m.%Y}\n"
            "#\n"
            f"{'Jahr':>4}  {'Temp_°C':>8}  {'Abw_K':>6}  {'Rang':>4}  "
            f"{'Tage':>4}  {'letzter_Tag':>11}\n")
    for year, row in annual.iterrows():
        f.write(f"{year:>4}  {row['temp']:>8.3f}  {row['anomaly']:>+6.3f}  "
                f"{row['rank']:>4d}  {row['days']:>4d}  "
                f"{row['last_date']:%d.%m.%Y}\n")
print(f"Jahreswerte gespeichert unter:\n{ANNUAL_FILE}")

# =========================================================
# KONSOLENAUSGABE
# =========================================================
print(f"\n{'=' * 60}\nJAHRESMITTEL GLOBAL – Abweichung zu {BASELINE}\n{'=' * 60}")
for year, row in annual.iterrows():
    note = f"  (bis {row['last_date']:%d.%m.})" if year == current_year and is_partial else ""
    print(f"{year}: {row['temp']:6.2f} °C | {row['anomaly']:+.2f} K | "
          f"Rang {row['rank']:2d}{note}")
print("\nTop 5 wärmste Jahre:")
for year, row in annual.nsmallest(5, "rank").iterrows():
    print(f"  {row['rank']}. {year}: {row['anomaly']:+.2f} K")
print("=" * 60)

# =========================================================
# LAUFENDES JAHR FÜR DIE GRAFIK AUS- ODER EINBLENDEN
# (TXT-Datei und Konsolenausgabe enthalten es weiterhin)
# =========================================================
if is_partial and not SHOW_CURRENT_YEAR:
    annual = annual.drop(current_year)
    annual["rank"] = annual["anomaly"].rank(ascending=False).astype(int)   # Rang nur unter vollen Jahren
    current_year = annual.index.max()
    last_date = annual.loc[current_year, "last_date"]
    is_partial = False

# =========================================================
# GRAFIK
# =========================================================
fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
box = dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
           edgecolor=FARBEN["box_rand"], alpha=0.9)

# Farben: blau (kalt) -> weiß -> rot (warm), Mittelpunkt bei 0 bzw. Mittel
# Symmetrisch um 0, damit gleiche Abweichungen gleich kräftig erscheinen
lim = annual["anomaly"].abs().max()
norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
cmap = plt.get_cmap("RdBu_r")
colors = cmap(norm(annual["anomaly"]))

bars = ax.bar(annual.index, annual["anomaly"], width=0.8, color=colors,
              edgecolor="gray", linewidth=0.4, zorder=3)

# Aktuelles (unvollständiges) Jahr kennzeichnen
if is_partial:
    bars[-1].set_hatch("///")
    if not LAUFEND_FARBIG:
        bars[-1].set_facecolor(FARBEN["bg"])
    bars[-1].set_edgecolor(FARBEN["fg"])
    bars[-1].set_linewidth(1.2)

# Werte im Balken an der Nulllinie (wie bei Monaten und Jahreszeiten):
# positive Werte knapp über 0, negative knapp unter 0
if SHOW_VALUES:
    for year, row in annual.iterrows():
        val = row["anomaly"]
        r_, g_, b_, _ = cmap(norm(val))
        laufend_weiss = is_partial and year == current_year and not LAUFEND_FARBIG
        dark = 0.299 * r_ + 0.587 * g_ + 0.114 * b_ < 0.6 and not laufend_weiss
        im_balken = abs(val) > 0.15 and not laufend_weiss     # Zahl passt ganz in den Balken
        if dark and im_balken:
            farbe = "white"                                   # dunkler Balken -> weiße Schrift
        elif DUNKEL and im_balken:
            farbe = "#3a3a40"                                 # heller Balken auf dunkler Grafik
        elif is_partial and year == current_year:
            farbe = FARBEN["fg"]
        else:
            farbe = FARBEN["zahl"]                            # Zahl steht (teils) auf dem Hintergrund
        ax.text(year, 0.02 if val >= 0 else -0.02, f"{val:+.2f}",
                va="bottom" if val >= 0 else "top", ha="center", rotation=90,
                fontsize=8 * SCHRIFT, fontweight="bold", zorder=6,
                color=farbe,
                gid="laufend_zahl" if is_partial and year == current_year else None)

# Gleitendes Mittel (zentriert, nur vollständige Jahre)
if SHOW_RUNNING_MEAN:
    full = annual.drop(current_year) if is_partial else annual
    if TREND == "loess30":
        running = loess(full["anomaly"], 30)
    else:
        running = full["anomaly"].rolling(RUNNING_WINDOW, center=True).mean().dropna()
    x, y = running.index.to_numpy(float), running.to_numpy(float)

    # Glättung: weiche Kurve (Catmull-Rom-Spline) durch die Jahreswerte,
    # damit keine Knicke zwischen den Jahren entstehen
    if SMOOTH_LINE and len(x) > 2:
        px = np.concatenate([[2 * x[0] - x[1]], x, [2 * x[-1] - x[-2]]])
        py = np.concatenate([[2 * y[0] - y[1]], y, [2 * y[-1] - y[-2]]])
        t = np.linspace(0, 1, 20, endpoint=False)[:, None]
        xs, ys = [], []
        for p in ((px, xs), (py, ys)):
            p0, p1, p2, p3 = p[0][:-3], p[0][1:-2], p[0][2:-1], p[0][3:]
            seg = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                         + (3 * p1 - p0 - 3 * p2 + p3) * t**3)
            p[1].extend(seg.T.ravel())
        x, y = np.append(xs, x[-1]), np.append(ys, y[-1])

    ax.plot(x, y, color=FARBEN["fg"], lw=2.2, ls="--",
            path_effects=[pe.Stroke(linewidth=4, foreground=FARBEN["linie_rand"]), pe.Normal()],
            label=("LOESS-Trend (30 Jahre)" if TREND == "loess30"
                       else f"{RUNNING_WINDOW}-jähriges gleitendes Mittel"), zorder=5)

# 1,5-K-Linie
if SHOW_1_5_LINE and BASELINE == "1850-1900":
    ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=1.8,
               label="1,5-K-Grenze (Pariser Klimaabkommen)", zorder=2)

ax.axhline(0, color=FARBEN["fg"], lw=0.8, zorder=2)

# Info-Kasten
cur = annual.loc[current_year]
full_years = annual.drop(current_year) if is_partial else annual
warmest = full_years["anomaly"].idxmax()        # Rekord nur aus vollständigen Jahren
info = (f"Temperaturmittel {current_year}"
        f"{f' (bis {last_date:%d.%m.})' if is_partial else ''}: {cur['temp']:.2f} °C\n"
        f"Temperaturabweichung {BASELINE}: {cur['anomaly']:+.2f} K  "
        f"({'bisher ' if is_partial else ''}Rang {cur['rank']})")
if not SHOW_RECORD:   # sonst steht das wärmste Jahr schon als Markierung in der Grafik
    info += f"\nWärmstes Jahr: {warmest} ({annual.loc[warmest, 'anomaly']:+.2f} K)"
info_text = ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
        fontsize=10 * SCHRIFT, fontweight="bold", bbox=box, zorder=10)
info_text.set_visible(SHOW_INFO)

# =========================================================
# ACHSEN, TITEL, LEGENDE
# =========================================================
ax.set_xlim(annual.index.min() - 1, annual.index.max() + 1)
ymin = min(annual["anomaly"].min(), 0)
ymax = max(annual["anomaly"].max(), 1.5 if SHOW_1_5_LINE and BASELINE == "1850-1900" else 0)
if Y_LIMITS:
    ax.set_ylim(*Y_LIMITS)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    outside = annual[(annual["anomaly"] < Y_LIMITS[0]) | (annual["anomaly"] > Y_LIMITS[1])]
    if not outside.empty:
        print(f"Hinweis: {len(outside)} Wert(e) liegen außerhalb der y-Achse "
              f"{Y_LIMITS}: {', '.join(str(y) for y in outside.index)}")
else:
    ax.set_ylim(min(ymin - 0.1, -0.27) if SHOW_VALUES else ymin - 0.15, ymax + 0.15)
ax.set_xticks(range((START_YEAR // 10) * 10, current_year + 1, 10))

# Überschrift in drei Zeilen wie bei den DWD-Grafiken:
# groß "Temperaturmittel", darunter fett "Global | Jahr", darunter normal der Bezug
zeile3 = ("Abweichung von der vorindustriellen Zeit (1850-1900)" if BASELINE == "1850-1900"
          else f"Abweichung vom Klimamittel {BASELINE}")
ax.set_title(zeile3, fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
mitte = ax.annotate("Global | Jahr", xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=16 * SCHRIFT, fontweight="bold")
ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=24 * SCHRIFT, fontweight="bold")
ax.set_xlabel("Jahr", fontsize=18 * SCHRIFT, fontweight="bold")
ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, axis="y", alpha=0.2, zorder=0)

fig.text(0.99, 0.01,
        "Datenquelle: Copernicus Climate Pulse (ERA5), C3S/ECMWF · CC BY 4.0\n"
        "https://pulse.climate.copernicus.eu/ · eigene Berechnung\n"
        "Visualisierung: JohnnyLauge.bsky.social",
        ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])

# Feste Ränder statt tight_layout -> gleiches Ergebnis auf jedem Rechner
# (oben etwas mehr Platz für die dreizeilige Überschrift)
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.87)

# Legende direkt unter dem Info-Kasten, links bündig mit ihm und mit demselben Abstand,
# den der Info-Kasten zum oberen Rand hat; das unvollständige Jahr steht mit drin
handles, labels = ax.get_legend_handles_labels()
if is_partial:
    handles.append(Patch(facecolor=colors[-1] if LAUFEND_FARBIG else FARBEN["bg"], edgecolor=FARBEN["fg"], hatch="///", linewidth=1.2))
    labels.append(f"{current_year} unvollständig (bis {last_date:%d.%m.})")
if handles:
    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.80),
                       frameon=True, fontsize=10 * SCHRIFT, fancybox=True,
                       framealpha=0.85, edgecolor=FARBEN["box_rand"],
                       facecolor=FARBEN["box_bg"], borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")


def legende_ausrichten():
    """Legende links bündig unter den Info-Kasten setzen (gleicher Abstand wie Info-Kasten zum
    oberen Rand) und, falls nötig, so weit nach unten, dass sie die 1,5-K-Linie nicht verdeckt."""
    legend = ax.get_legend()
    if legend is None:
        return
    for _ in range(2):                                    # zweiter Durchgang korrigiert Rundung
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        area = ax.get_window_extent(r)
        l = legend.get_frame().get_window_extent(r)           # sichtbarer Rahmen der Legende
        if SHOW_INFO:                                         # unter den Info-Kasten
            k = info_text.get_bbox_patch().get_window_extent(r)
            gap = area.y1 - k.y1                              # Abstand Info-Kasten zum oberen Rand
            ziel_oben, links = k.y0 - gap, k.x0
        else:                                                 # ohne Info-Kasten: oben links
            gap = 0.03 * area.height
            ziel_oben, links = area.y1 - gap, area.x0 + 0.01 * area.width
        if SHOW_1_5_LINE and BASELINE == "1850-1900":         # 1,5-K-Linie nicht verdecken
            linie = ax.transData.transform((0, 1.5))[1]
            ziel_oben = min(ziel_oben, linie - gap)
        anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        legend.set_bbox_to_anchor((anker.x0 + (links - l.x0) / area.width,
                                   anker.y0 + (ziel_oben - l.y1) / area.height),
                                  transform=ax.transAxes)


legende_ausrichten()

# Wärmstes Jahr markieren (Kasten mit Strich zum Balken, wie bei den DWD-Grafiken).
# Der Kasten darf keine Balken, den Info-Kasten oder die Legende verdecken: erst nach links,
# dann nach rechts versuchen und schrittweise anheben; reicht der Platz nicht, wird die
# y-Achse oben etwas erweitert.
if SHOW_RECORD:
    w_val = annual.loc[warmest, "anomaly"]

    def record_box(left, dy):
        return ax.annotate(f"Wärmstes Jahr {warmest}\n"
                           f"{annual.loc[warmest, 'temp']:.2f} °C | {w_val:+.2f} K",
                           xy=(warmest, w_val), xytext=(-50 if left else 50, dy),
                           textcoords="offset points", ha="right" if left else "left",
                           va="bottom", fontsize=10 * SCHRIFT, fontweight="bold", linespacing=1.3,
                           bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"]),
                           arrowprops=dict(arrowstyle="-", color=FARBEN["strich"], lw=1,
                                           connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                           relpos=(1, 0.5) if left else (0, 0.5),
                                           shrinkA=0, shrinkB=1),
                           zorder=9, gid="rekord")

    def is_free(m):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = m.get_bbox_patch().get_window_extent(r)
        area = ax.get_window_extent(r)
        if bb.x0 < area.x0 or bb.x1 > area.x1 or bb.y1 > area.y1 - 2:
            return False
        others = [t for t in ax.texts if t.get_bbox_patch() is not None and t is not m and t.get_visible()]
        if ax.get_legend():
            others.append(ax.get_legend())
        for o in others:
            ob = (o.get_bbox_patch() if hasattr(o, "get_bbox_patch") and o.get_bbox_patch() else o).get_window_extent(r)
            if bb.overlaps(ob):
                return False
        for b in bars:
            pb = b.get_window_extent(r)
            if bb.x0 < pb.x1 + 3 and bb.x1 > pb.x0 - 3 and bb.y0 < pb.y1 + 4 and bb.y1 > pb.y0 - 4:
                return False
        return True

    placed = None
    for _ in range(20):                                    # notfalls y-Achse oben erweitern
        for dy in range(10, 130, 6):
            for left in (True, False):
                m = record_box(left, dy)
                if is_free(m):
                    placed = m
                    break
                m.remove()
            if placed:
                break
        if placed:
            break
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.05 * (hi - lo))
    if placed is None:
        record_box(True, 10)
    legende_ausrichten()      # falls die y-Achse erweitert wurde, Legende neu ausrichten

# Immer gleiche Bildgröße: feste Figurgröße, KEIN bbox_inches="tight"
# (das würde das Bild je nach Überschrift/Text zuschneiden)
# Laufender weißer Balken: hinter seiner Zahl ein schmales weißes Feld, das nur die
# Schraffur ausblendet - der Rand des Balkens bleibt unberührt (erst jetzt, weil die
# y-Achse vorher noch erweitert werden kann)
zahl = next((t for t in ax.texts if t.get_gid() == "laufend_zahl"), None)
if zahl is not None and not LAUFEND_FARBIG:
    fig.canvas.draw()
    bb = zahl.get_window_extent(fig.canvas.get_renderer())
    (_, y0), (_, y1) = ax.transData.inverted().transform([(bb.x0, bb.y0 - 2), (bb.x1, bb.y1 + 2)])
    jahr = zahl.get_position()[0]
    ax.add_patch(Rectangle((jahr - 0.28, y0), 0.56, y1 - y0, facecolor=FARBEN["bg"],
                           edgecolor="none", zorder=5))   # Balken ist 0.8 breit -> Rand frei

# Lage des Info-Kastens merken (dort zeigt die Webseite die Werte unter der Maus)
info_rahmen = None
if SHOW_INFO:
    fig.canvas.draw()
    info_rahmen = info_text.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())

if SAVE_PNG:
    png_folder = os.path.join(FOLDER, "Grafiken")      # Unterordner für die Grafiken
    os.makedirs(png_folder, exist_ok=True)
    png_file = os.path.join(png_folder, PNG_NAME)
    if INFO_ALS_PLATZHALTER:
        # 1) volle Fassung zum Vergrößern und Herunterladen: mit Info- und Rekord-Kasten
        voll_file = os.path.splitext(png_file)[0] + "_voll.png"
        fig.savefig(voll_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
        fig.savefig(os.path.splitext(voll_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])   # PDF zum Herunterladen
        print(f"Grafik (mit Kästen) gespeichert unter:\n{voll_file}")
        # 2) interaktive Fassung: Kästen ausblenden, die Webseite zeigt die Werte selbst an
        info_text.set_visible(False)
        for t in ax.texts:
            if t.get_gid() == "rekord":
                t.set_visible(False)
    fig.savefig(png_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
    if not INFO_ALS_PLATZHALTER:                  # lokal: PDF neben dem PNG
        fig.savefig(os.path.splitext(png_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    print(f"Grafik gespeichert unter:\n{png_file}")
    # Lage der Balken für die Maus-Anzeige auf der Webseite (gleich für hell und dunkel)
    speichere_hover_daten(fig, ax, bars, annual, lambda j: str(j),
                          current_year if is_partial else None,
                          os.path.join(png_folder, "ERA5_global_jahreswerte" + ".json"),
                          info_rahmen=info_rahmen)

# Fenster nur lokal öffnen - auf GitHub (Umgebungsvariable CI ist gesetzt) gibt es keinen Bildschirm
if not os.environ.get("CI"):
    plt.show()
