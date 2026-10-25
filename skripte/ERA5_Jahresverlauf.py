"""
Globale Tagesmitteltemperatur (ERA5, Copernicus Climate Pulse) im Jahresverlauf:
das gewählte Jahr im Vergleich zu allen Jahren seit 1940, mit Klimamittel 1991-2020 und den
täglichen Wärme- und Kälterekorden. Neue Tagesrekorde werden zusätzlich in der Konsole aufgelistet.

Stil wie die übrigen Grafiken der Seite "Klima im Wandel" (Überschrift in drei Zeilen,
Info-Kasten oben links, Legende darunter, helle und dunkle Fassung).

Umgebungsvariablen (optional):
  ERA5_FOLDER            Datenordner (Standard: Climate_Pulse-Ordner auf dem PC)
  ERA5_OHNE_DOWNLOAD=1   vorhandene Tageswerte nutzen statt neu zu laden
  ERA5_DUNKEL=1          dunkle Fassung (..._dunkel.png)
  ERA5_WEB=1             Fassung für die Webseite (auf GitHub automatisch):
                           Grafiken/ERA5_global_verlauf_jahr_voll.png/.pdf  mit Info-Kasten (Vergrößern, Download)
                           Grafiken/ERA5_global_verlauf_jahr.png            ohne Kasten, für die Maus-Anzeige
                           Grafiken/ERA5_global_verlauf_jahr.json           Lage und Werte jedes Tages
"""

import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

# =========================================================
# EINSTELLUNGEN
# =========================================================
YEARS = [2026]                  # Hervorgehobene Jahre
CURRENT_YEAR_LINEWIDTH = 3.5    # Linienstärke der hervorgehobenen Jahre

SHOW_HISTORICAL = True          # Graue Hintergrundkurven
HISTORICAL_YEARS = (1940, None)  # None = automatisch bis zum Vorjahr der neuesten Daten

SHOW_1991_2020 = True           # Klimamittel 1991–2020

SHOW_RECORDS = True             # Tägliche Rekorde (Linien + Konsolenausgabe)
RECORD_YEARS = (1940, None)      # None = automatisch bis zum Vorjahr der neuesten Daten

SAVE_PNG = True                 # Grafik als PNG (und PDF) im Unterordner "Grafiken" speichern
PNG_DPI = 150                   # Auflösung: 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)           # Bildgröße in Zoll; 16:9 passt für Bluesky (wie die anderen Grafiken)
SCHRIFT = 1.3                   # Faktor für alle Schriftgrößen

# Helle oder dunkle Grafik. Lokal hier umstellen (oder ERA5_DUNKEL=1 setzen).
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"

# Farben je Jahr (hell / dunkel) - fehlt ein Jahr, gilt die Farbe des laufenden Jahres
YEAR_COLORS = {
    False: {2026: "#b2182b", 2025: "#ff3030", 2024: "#f5a623", 2003: "darkorange", 1976: "darkblue"},
    True:  {2026: "#ff3b4e", 2025: "#ff7f50", 2024: "#f5b841", 2003: "#ffa040", 1976: "#6aa6d8"},
}[DUNKEL]

# Farben wie bei den übrigen Grafiken
FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                fuss="gray", punkt_rand="white", hist="lightgray", hist_leg="darkgray",
                mittel="dimgray", warm="#d73027", kalt="#4575b4", gitter="#d0d0d4"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                fuss="#9a9aa2", punkt_rand="#1e1e21", hist="#8a8a92", hist_leg="#9a9aa2",
                mittel="#cfcfd4", warm="#ff6b6b", kalt="#6aa6d8", gitter="#4a4a50"),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"],
})

# Vorindustrielle Offsets je Monat (Januar bis Dezember), wie in den anderen ERA5-Skripten
MONTHLY_OFFSETS = dict(zip(range(1, 13), [
    0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93,
]))

MONTH_LABELS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
DREI = ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."]


def tag_monat(d):
    """Datum kurz wie auf der Webseite, z. B. '14. Okt.' (Mai ohne Punkt)."""
    return f"{d.day}. {DREI[d.month - 1]}"


URL = ("https://sites.ecmwf.int/data/climatepulse/data/series/"
       "era5_daily_series_2t_global.csv")
FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
FILE = os.path.join(FOLDER, "ERA5_global_tag.txt")              # Tageswerte (Original)
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")                   # Unterordner für Grafiken
QUELLE = ("Datenquelle: Copernicus Climate Pulse (ERA5), C3S/ECMWF · CC BY 4.0\n"
          "https://pulse.climate.copernicus.eu/ · eigene Berechnung\n"
          "Visualisierung: JohnnyLauge.bsky.social")

# =========================================================
# DOWNLOAD
# =========================================================
if os.environ.get("ERA5_OHNE_DOWNLOAD") == "1" and os.path.exists(FILE):
    print(f"Nutze vorhandene Datei:\n{FILE}")
else:
    print("Lade aktuelle ERA5-Daten herunter...")
    os.makedirs(FOLDER, exist_ok=True)
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with open(FILE, "wb") as f:
        f.write(response.content)
    print(f"Download abgeschlossen.\nDatei gespeichert unter:\n{FILE}")

# =========================================================
# DATEN AUFBEREITEN
# =========================================================
df = pd.read_csv(FILE, comment="#", parse_dates=["date"])
df["year"] = df["date"].dt.year
df["preindustrial_anomaly"] = (
    df["ano_91-20"] + df["date"].dt.month.map(MONTHLY_OFFSETS)
)

# Feste Jahresposition (1–365): Alle Jahre liegen auf derselben Achse.
# Der 29. Februar existiert im Referenzjahr 2001 nicht -> NaT -> wird entfernt.
df["pos"] = pd.to_datetime(
    "2001-" + df["date"].dt.strftime("%m-%d"), errors="coerce"
).dt.dayofyear
df = df.dropna(subset=["pos"]).astype({"pos": int})

# Tabelle: Zeilen = Jahresposition, Spalten = Jahre
temps = df.pivot(index="pos", columns="year", values="2t")

# Zeiträume "bis Vorjahr" automatisch bestimmen (z. B. 2027 -> bis 2026)
vorjahr = int(df["year"].max()) - 1
HISTORICAL_YEARS = (HISTORICAL_YEARS[0], HISTORICAL_YEARS[1] or vorjahr)
RECORD_YEARS = (RECORD_YEARS[0], RECORD_YEARS[1] or vorjahr)

# =========================================================
# GRAFIK
# =========================================================
fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
box = dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
           edgecolor=FARBEN["box_rand"], alpha=0.9)

# Historische Jahre
if SHOW_HISTORICAL:
    hist = [y for y in range(HISTORICAL_YEARS[0], HISTORICAL_YEARS[1] + 1)
            if y in temps.columns and y not in YEARS]
    lines = ax.plot(temps[hist], color=FARBEN["hist"], lw=0.8, alpha=0.25, zorder=1)
    if lines:
        lines[0].set_label(f"historische Daten seit {HISTORICAL_YEARS[0]}")

# Klimamittel 1991–2020
if SHOW_1991_2020:
    ax.plot(temps.loc[:, 1991:2020].mean(axis=1), color=FARBEN["mittel"], ls="--",
            lw=2.5, label="tägl. Klimamittel 1991-2020", zorder=3)

# Tägliche Rekorde
if SHOW_RECORDS:
    rec = temps.loc[:, RECORD_YEARS[0]:RECORD_YEARS[1]]
    warm, cold = rec.max(axis=1), rec.min(axis=1)
    warm_jahr, cold_jahr = rec.idxmax(axis=1), rec.idxmin(axis=1)
    ax.plot(warm, color=FARBEN["warm"], lw=0.8, alpha=0.85, zorder=2,
            label=f"tägl. Wärmerekord {RECORD_YEARS[0]}-{RECORD_YEARS[1]}")
    ax.plot(cold, color=FARBEN["kalt"], lw=0.8, alpha=0.85, zorder=2,
            label=f"tägl. Kälterekord {RECORD_YEARS[0]}-{RECORD_YEARS[1]}")

# Ausgewählte Jahre
current_year = max(YEARS)
farbe_jahr = lambda y: YEAR_COLORS.get(y, YEAR_COLORS.get(current_year, FARBEN["warm"]))
for year in YEARS:
    if year not in temps.columns:
        print(f"Keine Daten für {year} gefunden.")
        continue
    ax.plot(temps[year].dropna(), color=farbe_jahr(year),
            lw=CURRENT_YEAR_LINEWIDTH,
            label=f"tägl. Temperaturmittel {year}", zorder=4)

# Info-Kasten (oben links, wie bei den anderen Grafiken) und Markierung des letzten Tages
cur = df[df["year"] == current_year]
info_text = None

if not cur.empty:
    last = cur.iloc[-1]
    is_partial = last["date"] < pd.Timestamp(current_year, 12, 31)
    bis = f" (bis {tag_monat(last['date'])})" if is_partial else ""

    ax.scatter(last["pos"], last["2t"], s=70, color=farbe_jahr(current_year),
               edgecolor=FARBEN["punkt_rand"], linewidth=1.2, zorder=6)

    letzter_kasten = ax.annotate(
        f"{tag_monat(last['date'])} {last['date'].year}\n"
        f"{last['2t']:.2f} °C | {last['preindustrial_anomaly']:+.2f} K",
        xy=(last["pos"], last["2t"]), xytext=(10, 0), textcoords="offset points",
        ha="left", va="center", multialignment="right",   # beide Zeilen rechts bündig
        fontsize=12 * SCHRIFT, fontweight="bold",
        bbox={**box, "boxstyle": "round,pad=0.25"}, zorder=7,
    )

    # zwei Spalten: Beschriftung links, Zahlen rechtsbündig
    zeilen = [
        (f"Temperaturmittel {current_year}{bis}:", f"{cur['2t'].mean():.2f} °C"),
        ("Temperaturabweichung 1850-1900:", f"{cur['preindustrial_anomaly'].mean():+.2f} K"),
    ]
    stil = dict(fontsize=10 * SCHRIFT, fontweight="bold", linespacing=1.3, color=FARBEN["text"])
    spalten = HPacker(children=[
        TextArea("\n".join(z[0] for z in zeilen), textprops=stil),
        TextArea("\n".join(z[1] for z in zeilen), textprops=dict(stil, multialignment="right")),
    ], align="top", pad=0, sep=12)
    info_text = AnchoredOffsetbox(loc="upper left", child=spalten, pad=0.5, borderpad=0.8,
                                  frameon=True, bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
    info_text.patch.set_boxstyle("round,pad=0.3")
    info_text.patch.set(facecolor=FARBEN["box_bg"], edgecolor=FARBEN["box_rand"], alpha=0.9)
    info_text.set_zorder(10)
    ax.add_artist(info_text)

# =========================================================
# NEUE TAGESREKORDE (nur Konsole)
# =========================================================
if SHOW_RECORDS:
    for year in YEARS:
        d = df[df["year"] == year]
        if d.empty:
            continue

        print(f"\n{'=' * 75}\nREKORDAUSWERTUNG {year}\n"
              f"Vergleichszeitraum: {RECORD_YEARS[0]}–{RECORD_YEARS[1]}\n{'=' * 75}")

        for name, records, compare in (("Wärmerekorde", warm, d["2t"].gt),
                                       ("Kälterekorde", cold, d["2t"].lt)):
            old = d["pos"].map(records)
            new = d[compare(old)]

            print(f"\nNeue {name}: {len(new)}")
            for idx, row in new.iterrows():
                print(f"  {row['date']:%d.%m.%Y}: {row['2t']:.2f} °C\n"
                      f"      bisheriger Rekord: {old[idx]:.2f} °C\n"
                      f"      vorindustrielle Abweichung: "
                      f"{row['preindustrial_anomaly']:+.2f} K")

        print(f"\n{'=' * 75}")

# =========================================================
# ACHSEN, TITEL, LEGENDE
# =========================================================
month_starts = pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear
ax.set_xticks(month_starts)
ax.set_xticklabels(MONTH_LABELS)
ax.set_xlim(1, 365)
ax.set_ylim(10.5, 17.5)

# Überschrift in drei Zeilen wie bei den anderen Grafiken:
# groß "Temperaturmittel", darunter fett "Gebiet | Zeitraum | Datensatz", darunter normal der Vergleich
ax.set_title(f"Tageswerte im Vergleich zu den Jahren {HISTORICAL_YEARS[0]}-{HISTORICAL_YEARS[1]}",
             fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
mitte = ax.annotate(f"Global | {current_year} | ERA5", xy=(0.5, 1), xycoords=ax.title,
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=16 * SCHRIFT, fontweight="bold")
ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=24 * SCHRIFT, fontweight="bold")
ax.set_xlabel("Monat", fontsize=18 * SCHRIFT, fontweight="bold")
ax.set_ylabel("Temperatur [°C]", fontsize=18 * SCHRIFT, fontweight="bold")
ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
ax.grid(True, color=FARBEN["gitter"], alpha=0.5 if DUNKEL else 0.6, lw=0.8)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, QUELLE, ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])

# Feste Ränder statt tight_layout -> immer gleiche Bildgröße und Position
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.87)

# Legende links bündig direkt unter dem Info-Kasten (gleicher Abstand wie Info-Kasten zum Rand)
legend = ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80), borderaxespad=0,
                   frameon=True, fontsize=10 * SCHRIFT, fancybox=True,
                   framealpha=0.85, edgecolor=FARBEN["box_rand"], facecolor=FARBEN["box_bg"],
                   labelcolor=FARBEN["text"], borderpad=0.6)
plt.setp(legend.get_texts(), fontweight="bold")
# Legenden-Linie der historischen Daten kräftiger, sonst ist sie kaum zu sehen
for linie, text in zip(legend.get_lines(), legend.get_texts()):
    if text.get_text().startswith("historische"):
        linie.set_color(FARBEN["hist_leg"])
        linie.set_alpha(1)
        linie.set_linewidth(2)
if info_text is not None:
    # Beide Kästen gleich breit und links wie rechts bündig: Ist die Legende breiter, wird der
    # Info-Kasten verbreitert (mehr Abstand zwischen Beschriftung und Zahlen); ist sie schmaler,
    # bekommt sie einen eigenen, breiteren Rahmen. Der Inhalt bleibt links ausgerichtet.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = info_text.patch.get_window_extent(r)
    l = legend.get_frame().get_window_extent(r)
    if l.width > k.width:
        spalten.sep += (l.width - k.width) * 72 / fig.dpi
    for _ in range(2):                                    # zweiter Durchgang korrigiert Rundung
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        area = ax.get_window_extent(r)
        k = info_text.patch.get_window_extent(r)          # sichtbarer Rahmen des Info-Kastens
        l = legend.get_frame().get_window_extent(r)
        gap = area.y1 - k.y1
        anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        legend.set_bbox_to_anchor((anker.x0 + (k.x0 - l.x0) / area.width,
                                   anker.y0 + ((k.y0 - gap) - l.y1) / area.height),
                                  transform=ax.transAxes)
    # Rahmen der Legende auf die Breite des Info-Kastens ziehen, Inhalt bleibt links
    if k.width > l.width + 1:
        legend._mode = "expand"                           # Rahmen auf volle Breite des Ankers
        legend._legend_box.align = "left"                 # Einträge trotzdem links ausrichten
        x0, oben = (k.x0 - area.x0) / area.width, (l.y1 - area.y0) / area.height
        legend.set_bbox_to_anchor((x0, oben - 0.5, k.width / area.width, 0.5),
                                  transform=ax.transAxes)

# Grafik im Unterordner "Grafiken" speichern (PNG und PDF, hell bzw. dunkel)
komma = lambda v, plus=False: (f"{v:+.2f}" if plus else f"{v:.2f}").replace(".", ",").replace("-", "−")


def speichere_hover(datei):
    """Für die Webseite: je Kalendertag ein Feld über die ganze Höhe des Diagramms mit den Werten
    (gleiches Format wie bei den anderen Grafiken; 'text' steht fertig im Kasten)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    breite, hoehe = fig.canvas.get_width_height()
    fx = lambda v: round(min(max(v / breite, 0), 1), 5)
    fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
    achse = ax.get_window_extent(r)
    zu_px = ax.transData.transform
    mittel = temps.loc[:, 1991:2020].mean(axis=1)
    akt = cur.set_index("pos")
    tage = []
    for pos in range(1, 366):
        datum = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=pos - 1)
        x0 = zu_px((pos - 0.5, 0))[0]
        x1 = zu_px((pos + 0.5, 0))[0]
        oben, unten = zu_px((pos, warm[pos]))[1], zu_px((pos, cold[pos]))[1]
        rekorde = (f"Wärmerekord {komma(warm[pos])} °C ({warm_jahr[pos]}) | "
                   f"Kälterekord {komma(cold[pos])} °C ({cold_jahr[pos]})")
        if pos in akt.index:
            z = akt.loc[pos]
            neu = " · <strong>neuer Rekord</strong>" if z["2t"] > warm[pos] else ""
            text = (f"{tag_monat(z['date'])} {current_year}: {komma(z['2t'])} °C | "
                    f"{komma(z['preindustrial_anomaly'], True)} K{neu}<br>{rekorde}")
        else:
            text = (f"{tag_monat(datum)}: Klimamittel 1991-2020 {komma(mittel[pos])} °C<br>{rekorde}")
        tage.append(dict(text=text, x0=fx(x0), x1=fx(x1), y0=fy(oben), y1=fy(unten)))
    neuester = int(akt.index.max()) - 1 if len(akt) else len(tage) - 1
    info = None
    if info_text is not None:
        k = info_text.patch.get_window_extent(r)
        info = dict(x0=fx(k.x0), x1=fx(k.x1), y0=fy(k.y1), y1=fy(k.y0))
    with open(datei, "w", encoding="utf-8") as fh:
        json.dump(dict(achse=dict(x0=fx(achse.x0), x1=fx(achse.x1), y0=fy(achse.y1), y1=fy(achse.y0)),
                       info=info, neuester=neuester, balken=tage), fh, ensure_ascii=False)


if SAVE_PNG:
    os.makedirs(PNG_FOLDER, exist_ok=True)
    zusatz = "_dunkel" if DUNKEL else ""
    if WEB:
        basis = os.path.join(PNG_FOLDER, f"ERA5_global_verlauf_jahr{zusatz}")
        # 1) volle Fassung zum Vergrößern und Herunterladen
        fig.savefig(basis + "_voll.png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        fig.savefig(basis + "_voll.pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        # 2) Maus-Fassung: Info-Kasten und Kasten am letzten Tag ausblenden, die Seite zeigt die Werte
        if SHOW_RECORDS and info_text is not None:
            speichere_hover(os.path.join(PNG_FOLDER, "ERA5_global_verlauf_jahr.json"))
            info_text.set_visible(False)
            letzter_kasten.set_visible(False)
        fig.savefig(basis + ".png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        print(f"Grafik gespeichert unter:\n{basis}.png")
    else:
        png_file = os.path.join(PNG_FOLDER, f"ERA5_global_tag_{current_year}{zusatz}.png")
        fig.savefig(png_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
        fig.savefig(os.path.splitext(png_file)[0] + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        print(f"Grafik gespeichert unter:\n{png_file}")

if not os.environ.get("CI"):          # lokal anzeigen, auf GitHub nicht
    plt.show()
