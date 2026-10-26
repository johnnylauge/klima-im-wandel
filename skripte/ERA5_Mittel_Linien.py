"""
Globale Mitteltemperatur (ERA5) als Liniendiagramm, ab 1940: Abweichung von 1850-1900 in K
(wie Balken und Vergleich). Je Jahr das Mittel des Zeitraums, verbunden zu einer Linie, dazu ein LOESS-Trend.

Wahlweise für das ganze Jahr, eine Jahreszeit oder einen Monat:
  Jahr      Jahresmittel
  Frühling  März-Mai, Sommer Juni-August, Herbst September-November,
  Winter    Dezember-Februar (zählt zum Jahr von Januar/Februar, z. B. Winter 2025/26)
  Monat     z. B. Oktobermittel jedes Jahres

Der laufende (unvollständige) Zeitraum wird mit seiner bisherigen Abweichung gestrichelt
angehängt. Die Abweichung ist schon früh mit den vollen Zeiträumen vergleichbar; sein °C-Wert
(Rekord-Kasten, Maus-Anzeige) wird über die Abweichung aufs ganze Jahr umgerechnet.

Läuft lokal (Windows) und auf GitHub:
  - Lokal gelten die Einstellungen unten; die Grafik wird im Fenster gezeigt.
  - Auf GitHub gibt die Steuerdatei über Umgebungsvariablen vor:
      ERA5_FOLDER           Datenordner im Projekt
      ERA5_LINIEN           z. B. "alle" (überschreibt LINIEN_WAHL)
      ERA5_DUNKEL=1         dunkle Fassung (..._dunkel.png)
      ERA5_OHNE_DOWNLOAD=1  vorhandene Tageswerte nutzen statt neu zu laden
    Dort öffnet sich kein Fenster (kein Bildschirm), die Grafiken werden nur gespeichert.
"""

import os
import region                   # global oder Europa (region.py)
import json
import time
import requests
import numpy as np


def tag_monat(d):
    """Datum kurz wie auf der Webseite, z. B. '14. Okt.' (Mai ohne Punkt)."""
    return f"{d.day}. " + ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.",
                           "Sep.", "Okt.", "Nov.", "Dez."][d.month - 1]

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MultipleLocator


# Achsenbeschriftung in deutscher Schreibweise (1,5 statt 1.5) - gilt für alle Achsen ohne eigene Formatierung
from matplotlib.ticker import ScalarFormatter as _SF
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
START_YEAR = 1940

LINIEN_WAHL = "Jahr"            # "Jahr", eine Jahreszeit ("Herbst"), ein Monat ("Oktober" oder 10),
                                # eine Liste wie ["Jahr", "Sommer", 7] oder "alle" = alle 17 Grafiken
SHOW_CURRENT_YEAR = True        # laufenden Zeitraum (umgerechnet, offener Punkt) zeigen
SHOW_TREND = True               # Trendlinie
TREND = "loess30"               # "loess30" = LOESS-Glättung über 30 Jahre, "5jahre" = gleitendes Mittel
RUNNING_WINDOW = 5              # Fensterbreite in Jahren (nur für TREND = "5jahre")
SHOW_1_5_LINE = True            # 1,5-K-Grenze (über 1850-1900)
SHOW_RECORD = True              # wärmsten (vollständigen) Zeitraum markieren
SHOW_INFO = True                # Info-Kasten oben links (aktueller Wert und Rang)
SHOW_VALUES = False             # Temperatur als Zahl an jedem Jahr (lokal zum Posten)
LINIE = 3.6                     # Stärke der farbigen Linie
Y_TICK_STEP = None              # Abstand der y-Beschriftung; None = automatisch (0,25 oder 0,5)

SAVE_PNG = True                 # Grafik als PNG im Unterordner "Grafiken" speichern
PNG_DPI = 150                   # 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)           # 16:9 für Bluesky (wie die anderen Grafiken)
SCHRIFT = 1.3                   # Faktor für alle Schriftgrößen

# Helle oder dunkle Grafik. Lokal hier umstellen; auf GitHub setzt die Steuerdatei
# ERA5_DUNKEL=1 und erzeugt so zusätzlich eine dunkle Fassung für die Webseite.
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"

# Webseiten-Fassung (auf GitHub automatisch): ohne Zahlen. Es entstehen zwei Bilder:
#   ..._voll.png  mit Info-Kasten und Rekord-Kasten (zum Vergrößern und Herunterladen)
#   ....png       ohne die Kästen für die interaktive Ansicht
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
INFO_ALS_PLATZHALTER = SHOW_INFO and WEB
if WEB:
    SHOW_VALUES = False
if os.environ.get("ERA5_TREND"):
    TREND = os.environ["ERA5_TREND"]
if os.environ.get("ERA5_LINIEN"):
    LINIEN_WAHL = os.environ["ERA5_LINIEN"]

AUF_GITHUB = bool(os.environ.get("CI"))     # dort gibt es keinen Bildschirm

# Farben für helle und dunkle Grafik (wie bei den anderen Diagrammen)
FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                zahl="dimgray", fuss="gray", linie_rand="white", strich="dimgray", limit="#b2182b",
                linie="#6b6b70", punkt_rand="#555555"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                zahl="#b9b9c0", fuss="#9a9aa2", linie_rand="#1e1e21", strich="#9a9aa2", limit="#ff6b6b",
                linie="#9a9aa2", punkt_rand="#c8c8cc"),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"],
})

# Vorindustrielle Offsets je Monat (Januar bis Dezember)
MONTHLY_OFFSETS = dict(zip(range(1, 13), [
    0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93,
]))

SEASONS = {
    "Frühling": ([3, 4, 5], "fruehling"),
    "Sommer":   ([6, 7, 8], "sommer"),
    "Herbst":   ([9, 10, 11], "herbst"),
    "Winter":   ([12, 1, 2], "winter"),
}
MONTH_NAMES = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
               "August", "September", "Oktober", "November", "Dezember"]

URL = ("https://sites.ecmwf.int/data/climatepulse/data/series/"
       "era5_daily_series_2t_global.csv")
FOLDER = os.environ.get("ERA5_FOLDER",
                        r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
FILE = os.path.join(FOLDER, region.TAG_DATEI)                  # Tageswerte (global: Original)
if region.EUROPA:            # Europa: Abweichung schon zum gewählten Bezug, kein 1850-1900
    MONTHLY_OFFSETS = {m: 0.0 for m in range(1, 13)}
    SHOW_1_5_LINE = False
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")


def komma(x, stellen=2):
    """Zahl mit deutschem Komma, z. B. 15.123 -> '15,12'."""
    return f"{x:.{stellen}f}".replace(".", ",")


def k_text(x):
    """Abweichung mit Vorzeichen, z. B. '+1,60 K'."""
    return ("+" if x >= 0 else "−") + komma(abs(x)) + " K"


def loess(reihe, fenster=30):
    """LOESS-Glättung: für jedes Jahr eine lokale Gerade durch die 'fenster' nächstgelegenen
    Jahre, nahe Jahre zählen stärker (Tricube-Gewichte)."""
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
    """Weiche Kurve (Catmull-Rom-Spline) durch die Jahreswerte, ohne Knicke."""
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
if region.EUROPA or (os.environ.get("ERA5_OHNE_DOWNLOAD") == "1" and os.path.exists(FILE)):
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

ALLE_TAGE = region.anpassen_tage(pd.read_csv(FILE, comment="#", parse_dates=["date"]))
ALLE_TAGE["ano_pre"] = ALLE_TAGE["ano_91-20"] + ALLE_TAGE["date"].dt.month.map(MONTHLY_OFFSETS)


def zeitraum(wahl):
    """Übersetzt die Wahl in (Art, Name, Monate, Dateiname)."""
    if str(wahl).lower() == "jahr":
        return "jahr", "Jahr", list(range(1, 13)), f"{region.STAMM}_linien_jahr"
    for name, (monate, datei) in SEASONS.items():
        if str(wahl).lower() in (name.lower(), datei):
            return "saison", name, monate, f"{region.STAMM}_linien_saison_{datei}"
    if str(wahl).isdigit() and 1 <= int(wahl) <= 12:
        m = int(wahl)
    elif str(wahl).capitalize() in MONTH_NAMES:
        m = MONTH_NAMES.index(str(wahl).capitalize()) + 1
    else:
        raise ValueError(f"Unbekannte Auswahl '{wahl}'. Möglich: Jahr, "
                         f"{', '.join(SEASONS)}, Monatsname oder 1-12")
    return "monat", MONTH_NAMES[m - 1], [m], f"{region.STAMM}_linien_monat_{m:02d}"


def erstelle(wahl, anzeigen=True):
    """Zeichnet ein Liniendiagramm der Mitteltemperatur (Jahr, Jahreszeit oder Monat)."""
    art, name, monate, NAME = zeitraum(wahl)
    PNG_NAME = NAME + ("_dunkel" if DUNKEL else "") + ".png"

    def label(jahr):
        """Bezeichnung des Zeitraums, z. B. '2026', 'Sommer 2026', 'Winter 2025/26', 'März 2026'."""
        if art == "jahr":
            return str(jahr)
        if name == "Winter":
            return f"Winter {jahr - 1}/{str(jahr)[-2:]}"
        return f"{name} {jahr}"

    wort = {"jahr": "Jahr", "saison": name, "monat": name}[art]
    waermst = "Wärmstes" if art == "jahr" else "Wärmster"

    # =========================================================
    # DATEN AUFBEREITEN
    # =========================================================
    df = ALLE_TAGE[ALLE_TAGE["date"].dt.month.isin(monate)].copy()
    df["year"] = df["date"].dt.year
    if name == "Winter":                                  # Dezember zählt zum Folgejahr
        df.loc[df["date"].dt.month == 12, "year"] += 1
    df = df[df["year"] >= START_YEAR]

    def erwartet(jahr):
        if name == "Winter":
            return (pd.Timestamp(jahr, 3, 1) - pd.Timestamp(jahr - 1, 12, 1)).days
        tage = pd.date_range(f"{jahr}-01-01", f"{jahr}-12-31")
        return int(tage.month.isin(monate).sum())

    werte = df.groupby("year").agg(temp=("2t", "mean"), anomaly=("ano_pre", "mean"),
                                   days=("2t", "size"), last_date=("date", "max"))
    werte["expected"] = [erwartet(j) for j in werte.index]
    current_year = werte.index.max()
    behalten = (werte["days"] >= werte["expected"]) | (werte.index == current_year)
    werte = werte[behalten].copy()
    is_partial = werte.loc[current_year, "days"] < werte.loc[current_year, "expected"]
    last_date = werte.loc[current_year, "last_date"]
    werte["rank"] = werte["anomaly"].rank(ascending=False).astype(int)   # 1 = wärmster Zeitraum

    voll = werte.drop(current_year) if is_partial else werte
    # Abstand zwischen Temperatur und Abweichung: bei vollen Zeiträumen (fast) gleich
    # -> damit lässt sich die 1,5-K-Grenze in °C angeben und der laufende Zeitraum umrechnen
    abstand = float((voll["temp"] - voll["anomaly"]).median())
    werte["plot"] = werte["anomaly"]                  # gezeichnet wird die Abweichung (K)
    werte["c"] = werte["temp"]                        # °C für Rekord-Kasten und Maus-Anzeige
    if is_partial:                                    # laufender Zeitraum: °C umgerechnet
        werte.loc[current_year, "c"] = werte.loc[current_year, "anomaly"] + abstand
    voll = werte.drop(current_year) if is_partial else werte

    print(f"\n{'=' * 60}\nTEMPERATURMITTEL GLOBAL – {name.upper()} (°C)\n{'=' * 60}")
    for jahr, z in werte.tail(5).iterrows():
        extra = (f"  (bisher {z['temp']:.2f} °C bis {z['last_date']:%d.%m.}, umgerechnet)"
                 if jahr == current_year and is_partial else "")
        print(f"{label(jahr)}: {z['c']:.2f} °C | {z['anomaly']:+.2f} K | Rang {z['rank']}{extra}")

    zeige_laufend = is_partial and SHOW_CURRENT_YEAR
    bis_tag = f"Tag {last_date.day}" if art == "monat" else tag_monat(last_date)   # Monatsgrafik: nur der Tag
    gezeigt = werte if (zeige_laufend or not is_partial) else voll

    # =========================================================
    # GRAFIK
    # =========================================================
    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    box = dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
               edgecolor=FARBEN["box_rand"], alpha=0.9)

    lim = werte["anomaly"].abs().max()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    # Eine durchgehende Linie, deren Farbe der Abweichung folgt (Blau-Rot wie bei den Balken).
    # Zwischen zwei Jahren wird in feinen Stücken überblendet, damit der Verlauf weich ist.
    # Ein schmaler grauer Rand darunter hält auch fast weiße Stücke auf hellem Grund sichtbar.
    def farbige_linie(daten, gestrichelt=False):
        x = daten.index.to_numpy(float)
        y = daten["plot"].to_numpy(float)
        a = daten["anomaly"].to_numpy(float)
        if len(x) < 2:
            return
        t = np.linspace(0, 1, 13)
        xs = np.concatenate([x[i] + (x[i + 1] - x[i]) * t[:-1] for i in range(len(x) - 1)] + [x[-1:]])
        ys = np.interp(xs, x, y)
        an = np.interp(xs, x, a)
        punkte = np.column_stack([xs, ys])
        segmente = np.stack([punkte[:-1], punkte[1:]], axis=1)
        stil = (0, (1.2, 1.4)) if gestrichelt else "solid"
        ax.plot(xs, ys, color=FARBEN["linie"], lw=LINIE + 1.6, ls=stil, zorder=3,
                solid_capstyle="round", solid_joinstyle="round", dash_capstyle="round")
        ax.add_collection(LineCollection(segmente, colors=cmap(norm((an[:-1] + an[1:]) / 2)),
                                         linewidths=LINIE, linestyles=stil, capstyle="round",
                                         joinstyle="round", zorder=4))

    farbige_linie(voll)
    if zeige_laufend and len(voll):
        farbige_linie(werte.loc[[voll.index[-1], current_year]], gestrichelt=True)
    laufend_handle = None
    if zeige_laufend:
        laufend_handle = (Line2D([], [], color=cmap(norm(werte.loc[current_year, "anomaly"])), lw=LINIE,
                                 ls=(0, (1.2, 1.4)), dash_capstyle="round"),
                          f"{label(current_year)} unvollständig (bis {bis_tag})")

    # Zahlen an den Punkten (nur lokal, zum Posten)
    if SHOW_VALUES:
        for jahr, z in gezeigt.iterrows():
            ax.text(jahr, z["plot"] + 0.035, komma(z["plot"]), rotation=90, ha="center", va="bottom",
                    fontsize=8 * SCHRIFT, fontweight="bold", color=FARBEN["zahl"], zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground=FARBEN["bg"])])

    # Trend
    trend_linie = []
    if SHOW_TREND and len(voll) > 3:
        if TREND == "loess30":
            trend = loess(voll["plot"], 30)
            trend_name = "LOESS-Trend (30 Jahre)"
        else:
            trend = voll["plot"].rolling(RUNNING_WINDOW, center=True).mean().dropna()
            trend_name = f"{RUNNING_WINDOW}-jähriges gleitendes Mittel"
        x, y = weich(trend.index.to_numpy(float), trend.to_numpy(float))
        trend_linie = ax.plot(x, y, color=FARBEN["fg"], lw=2.2, ls="--",
                              path_effects=[pe.Stroke(linewidth=4, foreground=FARBEN["linie_rand"]),
                                            pe.Normal()],
                              label=trend_name, zorder=4)

    # 1,5 K über 1850-1900
    grenze = 1.5
    if SHOW_1_5_LINE:
        ax.axhline(grenze, color=FARBEN["limit"], ls=":", lw=1.8, zorder=2,
                   label="1,5-K-Grenze (Pariser Klimaabkommen)")
    ax.axhline(0, color=FARBEN["fg"], lw=0.8, zorder=2)

    # Info-Kasten
    cur = werte.loc[current_year]
    info = (f"Temperaturmittel {label(current_year)}"
            f"{f' (bis {bis_tag})' if is_partial else ''}: {region.temp_info(werte, current_year, is_partial)}\n"
            f"Temperaturabweichung {region.BEZUG}: {dz(cur['anomaly'], '+.2f')} K  "
            f"({'bisher ' if is_partial else ''}Rang {cur['rank']})")
    info_text = ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                        fontsize=10 * SCHRIFT, fontweight="bold", bbox=box, zorder=10)
    info_text.set_visible(SHOW_INFO)

    # =========================================================
    # ACHSEN, TITEL, LEGENDE
    # =========================================================
    ax.set_xlim(werte.index.min() - 1, werte.index.max() + 1)
    lo = gezeigt["plot"].min() - 0.15
    hi = max(gezeigt["plot"].max(), grenze if SHOW_1_5_LINE else -99) + 0.2
    ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
    schritt = Y_TICK_STEP or (0.25 if hi - lo < 2.5 else 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(schritt))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: komma(v)))
    ax.set_xticks(range((START_YEAR // 10) * 10, current_year + 1, 10))

    ax.set_title(region.UNTERTITEL, fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mitte = ax.annotate(f"{region.GEBIET} | {wort} | ERA5", xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=24 * SCHRIFT, fontweight="bold")
    # keine Beschriftung der x-Achse: Jahreszahlen bzw. Monate erklären sich selbst
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(True, alpha=0.2, zorder=0)

    fig.text(0.99, 0.01,
             region.FUSS + "\n"
             "Visualisierung: JohnnyLauge.bsky.social",
             ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.87)

    # Legende
    handles, labels = ax.get_legend_handles_labels()
    mittel_name = {"jahr": "Jahresmittel", "saison": "Jahreszeitenmittel", "monat": "Monatsmittel"}[art]
    class Verlauf(HandlerBase):
        """Legenden-Eintrag als kleine Linie mit Farbverlauf blau -> rot."""
        def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
            xs = np.linspace(-xd, -xd + w, 25)
            pts = np.column_stack([xs, np.full_like(xs, h / 2 - yd)])
            lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), colors=cmap(np.linspace(0.1, 0.95, 24)),
                                linewidths=LINIE, capstyle="butt", transform=trans)
            return [lc]

    verlauf = object()
    handles.insert(0, verlauf)
    labels.insert(0, f"{mittel_name} (Farbe nach Abweichung)")
    if laufend_handle:
        handles.append(laufend_handle[0])
        labels.append(laufend_handle[1])
    legend = ax.legend(handles, labels, handler_map={verlauf: Verlauf()},
                       loc="upper left", bbox_to_anchor=(0.01, 0.80),
                       frameon=True, fontsize=10 * SCHRIFT, fancybox=True,
                       framealpha=0.85, edgecolor=FARBEN["box_rand"],
                       facecolor=FARBEN["box_bg"], borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")
    legend.set_zorder(10)

    # Pixel von Punkten und Linien (Kästen sollen nichts davon verdecken)
    xy_punkte = gezeigt["plot"].reset_index().to_numpy(float)
    xy_linien = [xy_punkte] + [l.get_xydata() for l in trend_linie]
    if len(xy_punkte) > 1:      # Verbindungslinie fein abtasten, damit auch die Strecken zählen
        xs = np.linspace(xy_punkte[0, 0], xy_punkte[-1, 0], len(xy_punkte) * 6)
        xy_linien.append(np.column_stack([xs, np.interp(xs, xy_punkte[:, 0], xy_punkte[:, 1])]))

    def verdeckt(bb):
        for xy in xy_linien:
            p = ax.transData.transform(xy)
            if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                return True
        if SHOW_1_5_LINE:
            y15 = ax.transData.transform((0, grenze))[1]
            if bb.y0 < y15 < bb.y1:
                return True
        return False

    def legende_ausrichten():
        """Legende links bündig unter den Info-Kasten setzen; kreuzt sie dabei die 1,5-K-Linie,
        rutscht sie ganz darunter."""
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
            if SHOW_1_5_LINE:
                linie = ax.transData.transform((0, grenze))[1]
                if ziel_oben - l.height < linie + gap and ziel_oben > linie - gap:
                    ziel_oben = linie - gap
            anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
            legend.set_bbox_to_anchor((anker.x0 + (links - l.x0) / area.width,
                                       anker.y0 + (ziel_oben - l.y1) / area.height),
                                      transform=ax.transAxes)

    def kopf_rahmen():
        r = fig.canvas.get_renderer()
        rahmen = [legend.get_frame().get_window_extent(r).padded(4)]
        if SHOW_INFO:
            rahmen.append(info_text.get_bbox_patch().get_window_extent(r).padded(4))
        return rahmen

    def kopf_ueberdeckt():
        for bb in kopf_rahmen():
            for xy in xy_linien:
                p = ax.transData.transform(xy)
                if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                    return True
        return False

    # Platz oben links: notfalls die y-Achse unten/oben erweitern
    for _ in range(12):
        legende_ausrichten()
        if not kopf_ueberdeckt():
            break
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.05 * (hi_ - lo_))

    # Wärmsten (vollständigen) Zeitraum markieren: Kasten mit Zeiger wie bei den Balken
    if SHOW_RECORD and len(voll):
        rek = voll["anomaly"].idxmax()
        rz = voll.loc[rek]
        ax.scatter([rek], [rz["plot"]], s=150, facecolor="none", edgecolor=FARBEN["fg"],
                   linewidth=1.6, zorder=6, gid="rekord")

        def rekord_kasten(links, dy, dx):
            return ax.annotate(f"{waermst} {'Jahr ' if art == 'jahr' else ''}{label(rek)}\n"
                               f"{komma(rz['c'])} °C | {k_text(rz['anomaly'])}",
                               xy=(rek, rz["plot"]), xytext=(-dx if links else dx, dy),
                               textcoords="offset points", ha="right" if links else "left", va="center",
                               fontsize=10 * SCHRIFT, fontweight="bold", linespacing=1.3, gid="rekord",
                               bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"]),
                               arrowprops=dict(arrowstyle="-", color=FARBEN["strich"], lw=1,
                                               connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                               relpos=(1, 0.5) if links else (0, 0.5),
                                               shrinkA=0, shrinkB=7),
                               zorder=9)

        gesetzt = None
        for _ in range(15):
            legende_ausrichten()
            r = fig.canvas.get_renderer()
            belegt = kopf_rahmen()
            area = ax.get_window_extent(r)
            for dy in range(20, 300, 8):
                for dx in (40, 80, 130, 200):
                    for links in (True, False):
                        m = rekord_kasten(links, dy, dx)
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
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.05 * (hi_ - lo_))
        if gesetzt is None:
            rekord_kasten(True, 20, 40)
        legende_ausrichten()

    fig.canvas.draw()
    info_rahmen = info_text.get_bbox_patch().get_window_extent(fig.canvas.get_renderer()) if SHOW_INFO else None

    if SAVE_PNG:
        os.makedirs(PNG_FOLDER, exist_ok=True)
        png_file = os.path.join(PNG_FOLDER, PNG_NAME)
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
            speichere_hover_daten(fig, ax, gezeigt, label, current_year if is_partial else None,
                                  os.path.join(PNG_FOLDER, NAME + ".json"), info_rahmen)
    if anzeigen:
        plt.show()
    else:
        plt.close(fig)


def speichere_hover_daten(fig, ax, werte, beschriftung, laufend_jahr, datei, info_rahmen=None):
    """Schreibt für die Webseite, wo jeder Punkt im Bild liegt (Anteile 0..1, von oben links) und
    welche Werte dazugehören - gleiches Format wie bei den Balken, je Jahr ein kleines Feld um den Punkt."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    breite, hoehe = fig.canvas.get_width_height()
    fx = lambda v: round(min(max(v / breite, 0), 1), 5)
    fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
    achse = ax.get_window_extent(r)
    felder = []
    for jahr, z in werte.iterrows():
        px, py = ax.transData.transform((jahr, z["plot"]))
        x0, _ = ax.transData.transform((jahr - 0.4, 0))
        x1, _ = ax.transData.transform((jahr + 0.4, 0))
        felder.append(dict(
            jahr=int(jahr), label=beschriftung(jahr), rang=int(z["rank"]),
            # laufender Zeitraum: Wert wie im Bild (umgerechnet) -> ca=True, die Webseite zeigt "≈"
            temp=round(float(z["c"]), 3), abw=round(float(z["anomaly"]), 3),
            laufend=bool(jahr == laufend_jahr), ca=bool(jahr == laufend_jahr),
            x0=fx(min(x0, px - 9)), x1=fx(max(x1, px + 9)), y0=fy(py + 9), y1=fy(py - 9),
        ))
    with open(datei, "w", encoding="utf-8") as f:
        json.dump(dict(achse=dict(x0=fx(achse.x0), x1=fx(achse.x1),
                                  y0=fy(achse.y1), y1=fy(achse.y0)),
                       info=(dict(x0=fx(info_rahmen.x0), x1=fx(info_rahmen.x1),
                                  y0=fy(info_rahmen.y1), y1=fy(info_rahmen.y0))
                             if info_rahmen is not None else None),
                       balken=felder), f, ensure_ascii=False)
    print(f"Daten für die Maus-Anzeige gespeichert unter:\n{datei}")


# =========================================================
# AUSWAHL DURCHRECHNEN
# =========================================================
if str(LINIEN_WAHL).lower() == "alle":
    auswahl = ["Jahr"] + list(SEASONS) + list(range(1, 13))
elif isinstance(LINIEN_WAHL, (list, tuple)):
    auswahl = list(LINIEN_WAHL)
else:
    auswahl = [p.strip() for p in str(LINIEN_WAHL).split(",")]
for w in auswahl:
    zeitraum(w)                                     # Tippfehler früh melden
for w in auswahl:
    erstelle(w, anzeigen=len(auswahl) == 1 and not AUF_GITHUB)
