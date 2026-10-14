"""
Vergleich der Datensätze ERA5 (Copernicus), Berkeley Earth, NOAA und HadCRUT5 als Liniendiagramm:
Abweichung von 1850-1900 je Jahr, je Jahreszeit oder je Monat - eine Linie je Datensatz.

Das Skript lädt nichts herunter. Es liest die TXT-Dateien, die die anderen Skripte schreiben:
  ERA5_global_jahreswerte.txt, ERA5_global_monat_10.txt, ERA5_global_saison_herbst.txt ...
  BE_global_...txt   (BerkeleyEarth_Grafiken.py)
  NOAA_global_...txt (NOAA_Grafiken.py)
  HADCRUT_global_...txt (HadCRUT_Grafiken.py)
Fehlt eine Quelle, wird sie einfach weggelassen.

Der laufende (unvollständige) Zeitraum jeder Quelle wird gestrichelt angehängt. Die Quellen
enden unterschiedlich: ERA5 täglich, Berkeley Earth und NOAA etwa Mitte des Folgemonats.

Läuft lokal (Windows) und auf GitHub:
  - Auf GitHub liegen alle TXT-Dateien in ERA5_FOLDER; die Grafiken kommen nach ERA5_FOLDER/Grafiken.
  - Umgebungsvariablen: ERA5_FOLDER, BE_FOLDER, NOAA_FOLDER, VGL_FOLDER, VGL_WAHL, ERA5_DUNKEL=1
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# =========================================================
# EINSTELLUNGEN
# =========================================================
WAHL = "Jahr"                   # "Jahr", eine Jahreszeit, ein Monat (Name oder 1-12), Liste oder "alle"
SHOW_1_5_LINE = True
SHOW_INFO = True
LINIE = 2.6
X_TICK_STEP = 20

PNG_DPI = 150
GROESSE = (20, 11.25)
SCHRIFT = 1.3

DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
INFO_ALS_PLATZHALTER = SHOW_INFO and WEB
AUF_GITHUB = bool(os.environ.get("CI"))
if os.environ.get("VGL_WAHL"):
    WAHL = os.environ["VGL_WAHL"]

# Lokal: je Quelle ein eigener Ordner; auf GitHub alles in ERA5_FOLDER
BASIS = r"C:\Users\rafff\Documents\Wetterdaten"
ORDNER = {
    "ERA5": os.environ.get("ERA5_FOLDER", os.path.join(BASIS, "ERA5", "Climate_Pulse")),
    "BE": os.environ.get("BE_FOLDER") or os.environ.get("ERA5_FOLDER", os.path.join(BASIS, "Berkeley_Earth")),
    "NOAA": os.environ.get("NOAA_FOLDER") or os.environ.get("ERA5_FOLDER", os.path.join(BASIS, "NOAA")),
    "HADCRUT": os.environ.get("HADCRUT_FOLDER") or os.environ.get("ERA5_FOLDER", os.path.join(BASIS, "HadCRUT5")),
}
FOLDER = os.environ.get("VGL_FOLDER") or os.environ.get("ERA5_FOLDER", os.path.join(BASIS, "Vergleich"))
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")

FARBEN = {
    False: dict(bg="white",   fg="black",   text="#1d1d1f", box_bg="white",   box_rand="lightgray",
                fuss="gray", linie_rand="white", limit="#b2182b"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50",
                fuss="#9a9aa2", linie_rand="#1e1e21", limit="#ff6b6b"),
}[DUNKEL]
# Je Quelle eine eigene, gut unterscheidbare Farbe (auch bei Farbsehschwäche)
QUELLEN = {
    "ERA5": dict(name="ERA5 (Copernicus)", kurz="ERA5", farbe="#F5B841" if DUNKEL else "#E69F00"),
    "BE":   dict(name="Berkeley Earth",    kurz="Berkeley Earth", farbe="#56B4E9" if DUNKEL else "#0072B2"),
    "NOAA": dict(name="NOAA",              kurz="NOAA", farbe="#3FC79A" if DUNKEL else "#009E73"),
    "HADCRUT": dict(name="HadCRUT5 (Met Office)", kurz="HadCRUT5", farbe="#E59CC8" if DUNKEL else "#CC79A7"),
}
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"],
})

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


def zeitraum(wahl):
    """(Art, Name, Monate, Teil des TXT-Namens, Teil des Bildnamens)."""
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


def erwartete_tage(name, monate, jahr):
    if name == "Winter":
        return (pd.Timestamp(jahr, 3, 1) - pd.Timestamp(jahr - 1, 12, 1)).days
    return sum(pd.Period(f"{jahr}-{m:02d}").days_in_month for m in monate)


def lies_txt(datei):
    """TXT der anderen Skripte: Jahr Temp Abw Rang Tage letzter_Tag."""
    zeilen = [z.split() for z in open(datei, encoding="utf-8").read().splitlines()
              if z[:4].isdigit()]
    df = pd.DataFrame([(int(z[0]), float(z[2]), int(z[4]), z[5]) for z in zeilen],
                      columns=["jahr", "abw", "tage", "letzter"]).set_index("jahr")
    stand = next((z.replace("# Stand:", "").strip() for z in open(datei, encoding="utf-8")
                  if z.startswith("# Stand:")), "")
    return df, stand


def erstelle(wahl):
    art, name, monate, txt_teil, bild_teil = zeitraum(wahl)

    def label(jahr):
        if art == "jahr":
            return str(jahr)
        if name == "Winter":
            return f"Winter {jahr - 1}/{str(jahr)[-2:]}"
        return f"{name} {jahr}"

    reihen = {}
    for q in QUELLEN:
        datei = os.path.join(ORDNER[q], f"{q}_global_{txt_teil}.txt")
        if not os.path.exists(datei):
            print(f"Hinweis: {datei} fehlt - {q} wird weggelassen")
            continue
        df, stand = lies_txt(datei)
        letztes = df.index.max()
        df["laufend"] = False
        df.loc[letztes, "laufend"] = df.loc[letztes, "tage"] < erwartete_tage(name, monate, letztes)
        reihen[q] = (df, stand)
    if not reihen:
        print(f"Keine Daten für {name} - übersprungen")
        return

    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    ax.set_title("Abweichung von der vorindustriellen Zeit (1850-1900)",
                 fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mitte = ax.annotate(f"Global | {'Jahr' if art == 'jahr' else name} | Vergleich der Datensätze",
                        xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=24 * SCHRIFT, fontweight="bold")

    hindernisse = []
    handles, labels = [], []
    for q in ("HADCRUT", "BE", "NOAA", "ERA5"):      # ERA5 zuletzt, liegt obenauf
        if q not in reihen:
            continue
        df, _ = reihen[q]
        voll = df[~df["laufend"]]
        farbe = QUELLEN[q]["farbe"]
        ax.plot(voll.index, voll["abw"], color=farbe, lw=LINIE, zorder=4, solid_joinstyle="round",
                path_effects=[pe.Stroke(linewidth=LINIE + 1.6, foreground=FARBEN["linie_rand"]), pe.Normal()])
        if df["laufend"].any() and len(voll):
            j = df.index.max()
            ax.plot([voll.index[-1], j], [voll["abw"].iloc[-1], df.loc[j, "abw"]], color=farbe,
                    lw=LINIE, ls=(0, (1.2, 1.4)), dash_capstyle="round", zorder=4)
        hindernisse.append(df["abw"].reset_index().to_numpy(float))
        xs = np.linspace(df.index.min(), df.index.max(), len(df) * 6)
        hindernisse.append(np.column_stack([xs, np.interp(xs, df.index, df["abw"])]))
    for q in ("ERA5", "BE", "NOAA", "HADCRUT"):                 # Legende in fester Reihenfolge
        if q in reihen:
            handles.append(Line2D([], [], color=QUELLEN[q]["farbe"], lw=LINIE))
            labels.append(f"{QUELLEN[q]['name']} (ab {reihen[q][0].index.min()}, Stand {reihen[q][1]})")
    if any(df["laufend"].any() for df, _ in reihen.values()):
        handles.append(Line2D([], [], color=FARBEN["fg"], lw=LINIE, ls=(0, (1.2, 1.4)), dash_capstyle="round"))
        labels.append("gestrichelt: laufender Zeitraum, bisher")
    if SHOW_1_5_LINE:
        ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=1.8, zorder=2)
        handles.append(Line2D([], [], color=FARBEN["limit"], ls=":", lw=1.8))
        labels.append("1,5-K-Grenze (Pariser Klimaabkommen)")
    ax.axhline(0, color=FARBEN["fg"], lw=0.8, zorder=2)

    # Info-Kasten: neuester Zeitraum, Wert je Quelle
    neuestes = max(df.index.max() for df, _ in reihen.values())
    teile = []
    for q in ("ERA5", "BE", "NOAA", "HADCRUT"):
        if q in reihen:
            df, _ = reihen[q]
            if neuestes in df.index:
                teile.append(f"{QUELLEN[q]['kurz']} {k_text(df.loc[neuestes, 'abw'])}"
                             f"{' (bisher)' if df.loc[neuestes, 'laufend'] else ''}")
            else:
                teile.append(f"{QUELLEN[q]['kurz']} –")
    info = f"Temperaturabweichung {label(neuestes)}\n" + "  |  ".join(teile)
    info_text = ax.text(0.01, 0.97, info, transform=ax.transAxes, ha="left", va="top",
                        fontsize=10 * SCHRIFT, fontweight="bold", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=FARBEN["box_bg"],
                                  edgecolor=FARBEN["box_rand"], alpha=0.9))
    info_text.set_visible(SHOW_INFO)

    alle = pd.concat([df["abw"] for df, _ in reihen.values()])
    start = min(df.index.min() for df, _ in reihen.values())
    ax.set_xlim(start - 1, neuestes + 1)
    lo, hi = min(alle.min(), 0) - 0.1, max(alle.max(), 1.5) + 0.15
    ax.set_ylim(np.floor(lo * 4) / 4, np.ceil(hi * 4) / 4)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    erster_tick = start if start % X_TICK_STEP == 0 else (start // X_TICK_STEP + 1) * X_TICK_STEP
    ax.set_xticks(range(erster_tick, neuestes + 1, X_TICK_STEP))
    ax.set_xlabel("Jahr", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.grid(True, alpha=0.2, zorder=0)
    fig.text(0.99, 0.01, "Datenquellen: Copernicus Climate Pulse (ERA5), Berkeley Earth, NOAA NCEI, Met Office (HadCRUT5)\n"
             "Visualisierung: JohnnyLauge.bsky.social",
             ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.87)

    legend = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.80), frameon=True,
                       fontsize=10 * SCHRIFT, fancybox=True, framealpha=0.85, edgecolor=FARBEN["box_rand"],
                       facecolor=FARBEN["box_bg"], borderpad=0.6, borderaxespad=0)
    plt.setp(legend.get_texts(), fontweight="bold")
    legend.set_zorder(10)

    def ausrichten():
        """Legende links bündig unter den Info-Kasten; nicht über die 1,5-K-Linie."""
        for _ in range(2):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            area = ax.get_window_extent(r)
            l = legend.get_frame().get_window_extent(r)
            k = info_text.get_bbox_patch().get_window_extent(r)
            gap = area.y1 - k.y1
            ziel_oben, links = k.y0 - gap, k.x0
            if SHOW_1_5_LINE:
                linie = ax.transData.transform((0, 1.5))[1]
                if ziel_oben - l.height < linie + gap and ziel_oben > linie - gap:
                    ziel_oben = linie - gap
            anker = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
            legend.set_bbox_to_anchor((anker.x0 + (links - l.x0) / area.width,
                                       anker.y0 + (ziel_oben - l.y1) / area.height),
                                      transform=ax.transAxes)

    def verdeckt():
        r = fig.canvas.get_renderer()
        for bb in (legend.get_frame().get_window_extent(r).padded(4),
                   info_text.get_bbox_patch().get_window_extent(r).padded(4)):
            for xy in hindernisse:
                p = ax.transData.transform(xy)
                if ((p[:, 0] > bb.x0) & (p[:, 0] < bb.x1) & (p[:, 1] > bb.y0) & (p[:, 1] < bb.y1)).any():
                    return True
        return False

    for _ in range(12):
        ausrichten()
        if not verdeckt():
            break
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.05 * (hi_ - lo_))

    # Speichern: volle Fassung, interaktive Fassung, Daten für die Maus-Anzeige
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    info_rahmen = info_text.get_bbox_patch().get_window_extent(r)
    os.makedirs(PNG_FOLDER, exist_ok=True)
    basis = f"VGL_global_linien_{bild_teil}"
    png_file = os.path.join(PNG_FOLDER, basis + ("_dunkel" if DUNKEL else "") + ".png")
    if INFO_ALS_PLATZHALTER:
        fig.savefig(os.path.splitext(png_file)[0] + "_voll.png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
        info_text.set_visible(False)
    fig.savefig(png_file, dpi=PNG_DPI, facecolor=FARBEN["bg"])
    print(f"Grafik gespeichert unter:\n{png_file}")
    if WEB:
        fig.canvas.draw()
        breite, hoehe = fig.canvas.get_width_height()
        fx = lambda v: round(min(max(v / breite, 0), 1), 5)
        fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
        achse = ax.get_window_extent(r)
        felder = []
        for jahr in range(start, neuestes + 1):
            werte = []
            for q in ("ERA5", "BE", "NOAA", "HADCRUT"):
                if q in reihen and jahr in reihen[q][0].index:
                    z = reihen[q][0].loc[jahr]
                    werte.append(dict(q=QUELLEN[q]["kurz"], abw=round(float(z["abw"]), 3),
                                      laufend=bool(z["laufend"])))
            if not werte:
                continue
            ys = [ax.transData.transform((jahr, w["abw"]))[1] for w in werte]
            x0 = ax.transData.transform((jahr - 0.45, 0))[0]
            x1 = ax.transData.transform((jahr + 0.45, 0))[0]
            felder.append(dict(jahr=int(jahr), label=label(jahr), werte=werte,
                               x0=fx(x0), x1=fx(x1), y0=fy(max(ys) + 9), y1=fy(min(ys) - 9)))
        with open(os.path.join(PNG_FOLDER, basis + ".json"), "w", encoding="utf-8") as fh:
            json.dump(dict(achse=dict(x0=fx(achse.x0), x1=fx(achse.x1), y0=fy(achse.y1), y1=fy(achse.y0)),
                           info=dict(x0=fx(info_rahmen.x0), x1=fx(info_rahmen.x1),
                                     y0=fy(info_rahmen.y1), y1=fy(info_rahmen.y0)),
                           balken=felder), fh, ensure_ascii=False)
    if not ANZEIGEN:
        plt.close(fig)


if str(WAHL).lower() == "alle":
    auswahl = ["Jahr"] + list(SEASONS) + list(range(1, 13))
elif isinstance(WAHL, (list, tuple)):
    auswahl = list(WAHL)
else:
    auswahl = [p.strip() for p in str(WAHL).split(",")]
ANZEIGEN = len(auswahl) == 1 and not AUF_GITHUB
for w in auswahl:
    erstelle(w)
if ANZEIGEN:
    plt.show()
