"""
Globale Tagesmitteltemperatur (ERA5) als Punktdiagramm der Abweichung, ab 1940:
ein Punkt je Tag, alle Tage eines Jahres senkrecht übereinander.

Wahlweise für das ganze Jahr, eine Jahreszeit oder einen Monat:
  Jahr      alle Tage des Jahres
  Frühling  März-Mai, Sommer Juni-August, Herbst September-November,
  Winter    Dezember-Februar (zählt zum Jahr von Januar/Februar, z. B. Winter 2025/26)
  Monat     z. B. nur die Oktobertage jedes Jahres

Ablauf:
  1. Lädt die aktuellen Tageswerte der globalen 2-m-Temperatur (ERA5, Copernicus Climate
     Pulse) herunter und speichert sie als ERA5_global_tag.txt.
  2. Rechnet die Abweichung auf den vorindustriellen Bezug 1850-1900 um (Abweichung zu
     1991-2020 plus monatlicher Offset).
  3. Zeichnet je Tag einen Punkt in Blau-Rot nach Abweichung, dazu den Trend der Mittelwerte
     (LOESS 30 Jahre oder gleitendes Mittel), Hilfslinien, die 1,5-K-Linie, einen Info-Kasten
     oben links (aktuelles Mittel und Rang) mit Legende darunter sowie Kästen für das erste
     Überschreiten von 1,0 / 1,5 / 2,0 K und den höchsten Tageswert.
  4. Speichert die Grafik als PNG im Unterordner "Grafiken".

Läuft lokal (Windows) und auf GitHub:
  - Lokal gelten die Einstellungen unten; die Grafik wird im Fenster gezeigt.
  - Auf GitHub gibt die Steuerdatei über Umgebungsvariablen vor:
      ERA5_FOLDER           Datenordner im Projekt
      ERA5_PUNKTE           z. B. "alle" (überschreibt PUNKTE_WAHL)
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
from matplotlib.ticker import MultipleLocator

# =========================================================
# EINSTELLUNGEN
# =========================================================
START_YEAR = 1940

PUNKTE_WAHL = "Jahr"            # "Jahr", eine Jahreszeit ("Herbst"), ein Monat ("Oktober" oder 10),
                                # eine Liste wie ["Jahr", "Sommer", 7] oder "alle" = alle 17 Grafiken
THRESHOLDS = [1.0, 1.5, 2.0]    # Schwellen: erstes Überschreiten (Markierung + Konsole)
REFERENCE_LINES = [0.5, 1.0, 2.0]   # graue gestrichelte Hilfslinien (1,5 K kommt separat in Rot)
SHOW_1_5_LINE = True            # 1,5-K-Linie (Pariser Klimaabkommen)
SHOW_TREND = True               # Trendlinie der Mittelwerte (Jahr / Jahreszeit / Monat)
TREND = "loess30"               # "loess30" = LOESS-Glättung über 30 Jahre (Klimatrend),
                                # "5jahre"  = gleitendes Mittel über RUNNING_WINDOW Jahre
RUNNING_WINDOW = 5              # Fensterbreite in Jahren (nur für TREND = "5jahre")
SHOW_RECORD = True              # Kästen: erstes Überschreiten der Schwellen und höchster Tageswert
SHOW_INFO = True                # Info-Kasten oben links (aktuelles Mittel und Rang)
Y_TICK_STEP = 0.5               # Abstand der Beschriftung auf der y-Achse
PUNKT_GROESSE = {"jahr": 12, "saison": 14, "monat": 22}   # weniger Tage -> größere Punkte

SAVE_PNG = True                 # Grafik als PNG im Unterordner "Grafiken" speichern
PNG_DPI = 150                   # 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)           # 16:9 für Bluesky (wie die anderen Grafiken)
SCHRIFT = 1.3                   # Faktor für alle Schriftgrößen

# Helle oder dunkle Grafik. Lokal hier umstellen; auf GitHub setzt die Steuerdatei
# ERA5_DUNKEL=1 und erzeugt so zusätzlich eine dunkle Fassung für die Webseite.
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"

# Webseiten-Fassung (auf GitHub automatisch). Es entstehen zwei Bilder:
#   ..._voll.png  mit Info-Kasten und Rekord-Kästen (zum Vergrößern und Herunterladen)
#   ....png       ohne die Kästen für die interaktive Ansicht - dort zeigt die Webseite
#                 die Werte des Jahres unter der Maus an der Stelle des Info-Kastens.
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
INFO_ALS_PLATZHALTER = SHOW_INFO and WEB
if os.environ.get("ERA5_TREND"):
    TREND = os.environ["ERA5_TREND"]
if os.environ.get("ERA5_PUNKTE"):
    PUNKTE_WAHL = os.environ["ERA5_PUNKTE"]

AUF_GITHUB = bool(os.environ.get("CI"))     # dort gibt es keinen Bildschirm

# Farben für helle und dunkle Grafik (wie bei den Balkendiagrammen)
FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                fuss="gray", linie_rand="white", strich="dimgray", limit="#b2182b", hilfe="gray"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                fuss="#9a9aa2", linie_rand="#1e1e21", strich="#9a9aa2", limit="#ff6b6b", hilfe="#8a8a92"),
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

# Meteorologische Jahreszeiten: Monate und Dateiname
SEASONS = {
    "Frühling": ([3, 4, 5], "fruehling"),
    "Sommer":   ([6, 7, 8], "sommer"),
    "Herbst":   ([9, 10, 11], "herbst"),
    "Winter":   ([12, 1, 2], "winter"),
}
MONTH_NAMES = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
               "August", "September", "Oktober", "November", "Dezember"]
# Abgekürzte Monate mit Punkt; kurze Monatsnamen werden ausgeschrieben (ohne Punkt)
MONATE_KURZ = ["Jan.", "Feb.", "März", "Apr.", "Mai", "Juni",
               "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez."]

URL = ("https://sites.ecmwf.int/data/climatepulse/data/series/"
       "era5_daily_series_2t_global.csv")
# Lokal: Windows-Ordner. Auf GitHub setzt die Steuerdatei ERA5_FOLDER auf einen Ordner im Projekt.
FOLDER = os.environ.get("ERA5_FOLDER",
                        r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
FILE = os.path.join(FOLDER, region.TAG_DATEI)                  # Tageswerte (global: Original)
if region.EUROPA:            # Europa: Abweichung schon zum gewählten Bezug, kein 1850-1900
    MONTHLY_OFFSETS = {m: 0.0 for m in range(1, 13)}
    Y_TICK_STEP = 1.0              # Tageswerte streuen in Europa stark
    SHOW_1_5_LINE = False
    THRESHOLDS = []                # Schwellen 1/1,5/2 K beziehen sich auf 1850-1900
    REFERENCE_LINES = []
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")


def datum_kurz(d):
    """Datum mit abgekürztem Monat, z. B. '25. Feb. 1954' oder '15. Mai 1984'."""
    return f"{d.day}. {MONATE_KURZ[d.month - 1]} {d.year}"


def komma(x, stellen=1):
    """Zahl mit deutschem Komma, z. B. 1.5 -> '1,5'."""
    return f"{x:.{stellen}f}".replace(".", ",")


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
    """Übersetzt die Wahl in (Art, Name, Monate, Dateiname), z. B. ('monat', 'Oktober', [10], ...)."""
    if str(wahl).lower() == "jahr":
        return "jahr", "Jahr", list(range(1, 13)), f"{region.STAMM}_punkte_jahr"
    for name, (monate, datei) in SEASONS.items():
        if str(wahl).lower() in (name.lower(), datei):
            return "saison", name, monate, f"{region.STAMM}_punkte_saison_{datei}"
    if str(wahl).isdigit() and 1 <= int(wahl) <= 12:
        m = int(wahl)
    elif str(wahl).capitalize() in MONTH_NAMES:
        m = MONTH_NAMES.index(str(wahl).capitalize()) + 1
    else:
        raise ValueError(f"Unbekannte Auswahl '{wahl}'. Möglich: Jahr, "
                         f"{', '.join(SEASONS)}, Monatsname oder 1-12")
    return "monat", MONTH_NAMES[m - 1], [m], f"{region.STAMM}_punkte_monat_{m:02d}"


def erstelle(wahl, anzeigen=True):
    """Zeichnet ein Punktdiagramm (Jahr, Jahreszeit oder Monat)."""
    art, name, monate, NAME = zeitraum(wahl)
    PNG_NAME = NAME + ("_dunkel" if DUNKEL else "") + ".png"

    def label(jahr):
        """Bezeichnung des Zeitraums, z. B. '2026', 'Sommer 2026', 'Winter 2025/26', 'März 2026'."""
        if art == "jahr":
            return str(jahr)
        if name == "Winter":
            return f"Winter {jahr - 1}/{str(jahr)[-2:]}"
        return f"{name} {jahr}"

    # =========================================================
    # DATEN AUFBEREITEN
    # =========================================================
    df = ALLE_TAGE[ALLE_TAGE["date"].dt.month.isin(monate)].copy()
    df["year"] = df["date"].dt.year
    if name == "Winter":                                  # Dezember zählt zum Folgejahr
        df.loc[df["date"].dt.month == 12, "year"] += 1
    df = df[df["year"] >= START_YEAR]

    # erwartete Tage je Zeitraum -> unvollständige Zeiträume am Anfang entfernen
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
    werte = werte[behalten]
    df = df[df["year"].isin(werte.index)]
    is_partial = werte.loc[current_year, "days"] < werte.loc[current_year, "expected"]
    last_date = werte.loc[current_year, "last_date"]
    werte["rank"] = werte["anomaly"].rank(ascending=False).astype(int)   # 1 = wärmster Zeitraum

    # Erstes Überschreiten der Schwellen und höchster Tageswert
    first = {t: df.loc[df["ano_pre"] >= t].iloc[0] if (df["ano_pre"] >= t).any() else None
             for t in THRESHOLDS}
    peak = df.loc[df["ano_pre"].idxmax()]

    print(f"\n{'=' * 60}\nTAGESWERTE {region.GEBIET.upper()} – {name.upper()}\n{'=' * 60}")
    for t, row in first.items():
        if row is None:
            print(f"{komma(t)} K wurde noch nicht überschritten.")
        else:
            print(f"{komma(t)} K erstmals überschritten: {row['date']:%d.%m.%Y} | "
                  f"{row['2t']:.3f} °C | {row['ano_pre']:+.3f} K")
    print(f"Höchster Tageswert: {peak['date']:%d.%m.%Y} | {peak['ano_pre']:+.3f} K")

    # =========================================================
    # GRAFIK
    # =========================================================
    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    box = dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
               edgecolor=FARBEN["box_rand"], alpha=0.9)

    # Punkte: ein Punkt je Tag, Farbe nach Abweichung. Rote Seite reicht bis zum
    # 99-%-Wert (darüber volles Dunkelrot), blaue Seite wird für die wenigen kalten Tage gestreckt.
    farb_oben = df["ano_pre"].quantile(0.99)
    farb_unten = min(df["ano_pre"].min(), -0.3)
    punkte = ax.scatter(df["year"], df["ano_pre"], c=df["ano_pre"], cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=farb_unten, vcenter=0, vmax=farb_oben),
                        s=PUNKT_GROESSE[art], alpha=0.95, edgecolors="none", zorder=2,
                        rasterized=True)   # im PDF als Bild, sonst würde es mit 30 000 Punkten riesig

    # Trend der Mittelwerte (nur vollständige Zeiträume)
    trend_linie = []
    if SHOW_TREND:
        voll = werte.drop(current_year) if is_partial else werte
        if TREND == "loess30":
            trend = loess(voll["anomaly"], 30)
            trend_name = "LOESS-Trend (30 Jahre)"
        else:
            trend = voll["anomaly"].rolling(RUNNING_WINDOW, center=True).mean().dropna()
            trend_name = f"{RUNNING_WINDOW}-jähriges gleitendes Mittel"
        x, y = weich(trend.index.to_numpy(float), trend.to_numpy(float))
        trend_linie = ax.plot(x, y, color=FARBEN["fg"], lw=2.5, ls="--",
                              path_effects=[pe.Stroke(linewidth=4.5, foreground=FARBEN["linie_rand"]),
                                            pe.Normal()],
                              label=trend_name, zorder=4)

    # Hilfslinien
    ax.axhline(0, color=FARBEN["fg"], lw=1, zorder=1)
    for y in REFERENCE_LINES:
        ax.axhline(y, color=FARBEN["hilfe"], ls="--", lw=1, alpha=0.6, zorder=1)
    if SHOW_1_5_LINE:
        ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=2,
                   label="1,5-K-Grenze (Pariser Klimaabkommen)", zorder=3)

    # Markierte Tage: erstes Überschreiten der Schwellen und höchster Tageswert
    marken = [(f"Erstmals über {komma(t)} K", r) for t, r in first.items() if r is not None]
    hoch = f"Höchster Tageswert {komma(peak['ano_pre'], 2)} K"
    gleich = [i for i, (_, r) in enumerate(marken) if r["date"] == peak["date"]]
    if gleich:                                   # Höchstwert = erstes Überschreiten -> ein Kasten
        marken[gleich[0]] = (f"{hoch}\n{marken[gleich[0]][0].replace('Erstmals', 'erstmals')}", peak)
    else:
        marken.append((hoch, peak))
    if SHOW_RECORD:
        for _, row in marken:
            ax.scatter(row["year"], row["ano_pre"], s=PUNKT_GROESSE[art] * 5, facecolor="none",
                       edgecolor=FARBEN["fg"], linewidth=1.6, zorder=6, gid="rekord")

    # Info-Kasten
    cur = werte.loc[current_year]
    titel_info = "Temperaturmittel " + label(current_year)
    bis_tag = f"Tag {last_date.day}" if art == "monat" else tag_monat(last_date)   # Monatsgrafik: nur der Tag
    info = (f"{titel_info}{f' (bis {bis_tag})' if is_partial else ''}: {region.temp_info(werte, current_year, is_partial)}\n"
            f"Temperaturabweichung {region.BEZUG}: {cur['anomaly']:+.2f} K  "
            f"({'bisher ' if is_partial else ''}Rang {cur['rank']})")
    info_text = ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                        fontsize=10 * SCHRIFT, fontweight="bold", bbox=box, zorder=10)
    info_text.set_visible(SHOW_INFO)

    # =========================================================
    # ACHSEN, TITEL, LEGENDE
    # =========================================================
    ax.set_xlim(werte.index.min() - 1, werte.index.max() + 1)
    lo = min(df["ano_pre"].min(), 0) - 0.15
    hi = max(df["ano_pre"].max(), 1.5 if SHOW_1_5_LINE else 0) + 0.2
    ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    ax.set_xticks(range((START_YEAR // 10) * 10, current_year + 1, 10))

    ax.set_title(region.UNTERTITEL,
                 fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mitte_text = f"{region.GEBIET} | {name} | ERA5"          # name = "Jahr", Jahreszeit oder Monat
    mitte = ax.annotate(mitte_text, xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=24 * SCHRIFT, fontweight="bold")
    ax.set_xlabel("Jahr", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(True, alpha=0.2, zorder=0)

    fig.text(0.99, 0.01,
             region.FUSS + "\n"
             "Visualisierung: JohnnyLauge.bsky.social",
             ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])

    # Feste Ränder statt tight_layout -> immer gleiche Bildgröße und Position
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.87)

    # Legende: ein Punkt steht für einen Tag
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Line2D([], [], ls="none", marker="o", markersize=7,
                             markerfacecolor=plt.get_cmap("RdBu_r")(0.85), markeredgecolor="none"))
    labels.insert(0, "Tageswert (ein Punkt je Tag)")
    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.80),
                       frameon=True, fontsize=10 * SCHRIFT, fancybox=True,
                       framealpha=0.85, edgecolor=FARBEN["box_rand"],
                       facecolor=FARBEN["box_bg"], borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")
    legend.set_zorder(10)

    xy_punkte = df[["year", "ano_pre"]].to_numpy(float)

    def legende_ausrichten():
        """Legende links bündig unter den Info-Kasten setzen; kreuzt sie dabei die 1,5-K-Linie,
        rutscht sie ganz darunter."""
        for _ in range(2):                                    # zweiter Durchgang korrigiert Rundung
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
                linie = ax.transData.transform((0, 1.5))[1]
                if ziel_oben - l.height < linie + gap and ziel_oben > linie - gap:
                    ziel_oben = linie - gap
            anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
            legend.set_bbox_to_anchor((anker.x0 + (links - l.x0) / area.width,
                                       anker.y0 + (ziel_oben - l.y1) / area.height),
                                      transform=ax.transAxes)

    def kopf_ueberdeckt():
        """True, wenn Info-Kasten oder Legende Tagespunkte oder die Trendlinie verdecken."""
        r = fig.canvas.get_renderer()
        rahmen = [legend.get_frame().get_window_extent(r).padded(4)]
        if SHOW_INFO:
            rahmen.append(info_text.get_bbox_patch().get_window_extent(r).padded(4))
        pix = [ax.transData.transform(xy_punkte)]
        pix += [ax.transData.transform(l.get_xydata()) for l in trend_linie]
        for bb in rahmen:
            for p in pix:
                if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                    return True
        return False

    # Platz oben links: notfalls die y-Achse oben erweitern
    for _ in range(12):
        legende_ausrichten()
        if not kopf_ueberdeckt():
            break
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.05 * (hi_ - lo_))

    # =========================================================
    # KÄSTEN DER MARKIERTEN TAGE
    # Zeiger waagerecht aus dem Kasten heraus, dann senkrecht zum Punkt. Der Kasten wird
    # schrittweise angehoben und seitlich versetzt, bis er nichts Wichtiges verdeckt: keine
    # Tagespunkte, nicht die Trendlinie, nicht die 1,5-K-Linie, nicht Info-Kasten/Legende und
    # keinen anderen Kasten. Reicht der Platz nicht, wird die y-Achse oben erweitert.
    # =========================================================
    def rekord_kasten(text, row, links, dy, dx=60):
        return ax.annotate(f"{text}\n{datum_kurz(row['date'])}",
                           xy=(row["year"], row["ano_pre"]), xytext=(-dx if links else dx, dy),
                           textcoords="offset points", ha="right" if links else "left", va="center",
                           fontsize=10 * SCHRIFT, fontweight="bold", linespacing=1.3, gid="rekord",
                           bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"]),
                           arrowprops=dict(arrowstyle="-", color=FARBEN["strich"], lw=1,
                                           connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                           relpos=(1, 0.5) if links else (0, 0.5),
                                           shrinkA=0, shrinkB=4),
                           zorder=9)

    cache = {}

    def zeiger(kasten, r):
        bb = kasten.get_bbox_patch().get_window_extent(r)
        px, py = ax.transData.transform(kasten.xy)
        links = kasten.get_horizontalalignment() == "right"
        sx, sy = (bb.x1 if links else bb.x0), (bb.y0 + bb.y1) / 2
        return [((sx, sy), (px, sy)), ((px, sy), (px, py))]

    def kreuzt(strecke, rect):
        (x0, y0), (x1, y1) = strecke
        xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        return xa < rect.x1 and xb > rect.x0 and ya < rect.y1 and yb > rect.y0

    def frei(kasten, belegt, striche):
        r = fig.canvas.get_renderer()
        kasten.update_positions(r)
        kasten.update_bbox_position_size(r)
        bb = kasten.get_bbox_patch().get_window_extent(r).padded(4)
        area = ax.get_window_extent(r)
        if bb.x0 < area.x0 or bb.x1 > area.x1 or bb.y1 > area.y1 or bb.y0 < area.y0:
            return False
        if any(bb.overlaps(o) for o in belegt):
            return False
        if any(kreuzt(st, o) for st in zeiger(kasten, r) for o in belegt):
            return False
        if any(kreuzt(st, bb) for st in striche):
            return False
        if cache.get("ylim") != ax.get_ylim():
            cache["ylim"] = ax.get_ylim()
            cache["punkte"] = ax.transData.transform(xy_punkte)
            cache["linien"] = [ax.transData.transform(l.get_xydata()) for l in trend_linie]
        for p in [cache["punkte"]] + cache["linien"]:
            if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                return False
        if SHOW_1_5_LINE:
            y15 = ax.transData.transform((0, 1.5))[1]
            if bb.y0 < y15 < bb.y1:
                return False
        return True

    if SHOW_RECORD:
        jahr_mitte = (werte.index.min() + werte.index.max()) / 2
        for durchgang in range(12):
            for t in [t for t in ax.texts if t.get_gid() == "rekord"]:
                t.remove()
            legende_ausrichten()
            r = fig.canvas.get_renderer()
            belegt = [legend.get_frame().get_window_extent(r).padded(4)]
            if SHOW_INFO:
                belegt.append(info_text.get_bbox_patch().get_window_extent(r).padded(4))
            striche = []
            alle_platziert = True
            for text, row in marken:
                kasten = None
                standard = row["year"] > jahr_mitte          # rechts im Bild -> Kasten links
                for dy in range(14, 400, 8):
                    for dx in (30, 60, 100, 150, 220):
                        for links in (standard, not standard):
                            kasten = rekord_kasten(text, row, links, dy, dx)
                            if frei(kasten, belegt, striche):
                                break
                            kasten.remove()
                            kasten = None
                        if kasten is not None:
                            break
                    if kasten is not None:
                        break
                if kasten is None:
                    alle_platziert = False
                    if durchgang < 11:
                        break
                    kasten = rekord_kasten(text, row, standard, 14)   # letzter Versuch: trotzdem setzen
                belegt.append(kasten.get_bbox_patch().get_window_extent(r).padded(4))
                striche += zeiger(kasten, r)
            if alle_platziert:
                break
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ + 0.06 * (hi_ - lo_))
        legende_ausrichten()

    # Lage des Info-Kastens merken (dort zeigt die Webseite die Werte unter der Maus)
    fig.canvas.draw()
    info_rahmen = info_text.get_bbox_patch().get_window_extent(fig.canvas.get_renderer()) if SHOW_INFO else None

    if SAVE_PNG:
        os.makedirs(PNG_FOLDER, exist_ok=True)
        png_file = os.path.join(PNG_FOLDER, PNG_NAME)
        if INFO_ALS_PLATZHALTER:
            # 1) volle Fassung zum Vergrößern und Herunterladen: mit allen Kästen
            voll_file = os.path.splitext(png_file)[0] + "_voll.png"
            fig.savefig(voll_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
            fig.savefig(os.path.splitext(voll_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])   # PDF zum Herunterladen
            print(f"Grafik (mit Kästen) gespeichert unter:\n{voll_file}")
            # 2) interaktive Fassung: Kästen und Ringe ausblenden, die Webseite zeigt die Werte
            info_text.set_visible(False)
            for t in ax.texts + ax.collections:
                if t.get_gid() == "rekord":
                    t.set_visible(False)
        fig.savefig(png_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
        if not INFO_ALS_PLATZHALTER:                  # lokal: PDF neben dem PNG
            fig.savefig(os.path.splitext(png_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        print(f"Grafik gespeichert unter:\n{png_file}")
        if WEB:
            speichere_hover_daten(fig, ax, df, werte, label, current_year if is_partial else None,
                                  os.path.join(PNG_FOLDER, NAME + ".json"), info_rahmen)
    if anzeigen:
        plt.show()
    else:
        plt.close(fig)


def speichere_hover_daten(fig, ax, df, werte, beschriftung, laufend_jahr, datei, info_rahmen=None):
    """Schreibt für die Webseite, wo die Punkte jedes Jahres im Bild liegen (Anteile 0..1 von
    Breite/Höhe, gemessen von oben links) und welche Mittelwerte dazugehören. Gleiches Format
    wie bei den Balkendiagrammen: je Jahr ein schmales Feld vom tiefsten bis zum höchsten Punkt."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    breite, hoehe = fig.canvas.get_width_height()
    fx = lambda v: round(min(max(v / breite, 0), 1), 5)
    fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
    achse = ax.get_window_extent(r)
    spanne = df.groupby("year")["ano_pre"].agg(["min", "max"])
    felder = []
    for jahr, z in werte.iterrows():
        (x0, y0), (x1, y1) = ax.transData.transform([(jahr - 0.4, spanne.loc[jahr, "min"]),
                                                     (jahr + 0.4, spanne.loc[jahr, "max"])])
        felder.append(dict(
            jahr=int(jahr), label=beschriftung(jahr), rang=int(z["rank"]),
            **region.temp_hover(werte, jahr, laufend_jahr), abw=round(float(z["anomaly"]), 3),
            laufend=bool(jahr == laufend_jahr),
            x0=fx(x0), x1=fx(x1), y0=fy(y1 + 4), y1=fy(y0 - 4),   # oben / unten im Bild
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
if str(PUNKTE_WAHL).lower() == "alle":
    auswahl = ["Jahr"] + list(SEASONS) + list(range(1, 13))
elif isinstance(PUNKTE_WAHL, (list, tuple)):
    auswahl = list(PUNKTE_WAHL)
else:
    auswahl = [p.strip() for p in str(PUNKTE_WAHL).split(",")]
for w in auswahl:
    zeitraum(w)                                     # Tippfehler früh melden
for w in auswahl:
    # bei mehreren Grafiken (oder auf GitHub) nur speichern, nicht einzeln anzeigen
    erstelle(w, anzeigen=len(auswahl) == 1 and not AUF_GITHUB)
