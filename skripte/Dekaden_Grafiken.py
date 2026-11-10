"""
Dekaden-Grafiken für "Klima im Wandel" (Entwurf zum Feintunen auf dem PC).

Drei Darstellungen, je hell und dunkel:
  balken   Mittel je Dekade (Durchschnitt aus ERA5, NASA, NOAA, HadCRUT5, Berkeley Earth),
           heller Teil = Zuwachs zur Dekade davor, graue Linie = Spanne der fünf Datensätze
  glocken  Verteilung der ERA5-Tageswerte je Dekade (wie häufig welche Abweichung vorkam)
  punkte   ein Punkt je Jahr (Mittel aus fünf Datensätzen), linearer Trend innerhalb jeder Dekade

Daten:
  VGL_global_jahreswerte.txt  (Vergleich der fünf Datensätze; entsteht beim Vergleich auf GitHub)
  ERA5_global_tag.txt         (ERA5-Tageswerte)
Liegen sie nicht in ERA5_FOLDER, werden sie geladen: der Vergleich von der Webseite,
die Tageswerte von Copernicus Climate Pulse.

Grafiken: ERA5_FOLDER/Grafiken/Dekaden_<art>[_dekade][_dunkel].png (+ .pdf)

Umgebungsvariablen (für GitHub, auf dem PC einfach die EINSTELLUNGEN unten ändern):
  ERA5_FOLDER, ERA5_DUNKEL=1, DEKADEN_WAHL (z. B. "balken,glocken"), DEKADEN_FARBEN ("rot" oder "dekade")
"""

import os
import io
import requests
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("CI"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, rgb_to_hsv, hsv_to_rgb

# Achsenbeschriftung in deutscher Schreibweise (1,5 statt 1.5)
from matplotlib.ticker import ScalarFormatter as _SF
import angleichung      # Abstand 1991-2020 -> 1850-1900 wie Copernicus (je Tag)
import kopf             # einheitliche Überschrift
if not getattr(_SF, "_komma", False):
    _sf_alt = _SF.__call__
    _SF.__call__ = lambda self, x, pos=None: _sf_alt(self, x, pos).replace(".", ",")
    _SF._komma = True

# =========================================================
# EINSTELLUNGEN (zum Feintunen)
# =========================================================
WAHL = os.environ.get("DEKADEN_WAHL", "balken,glocken,punkte").split(",")
FARBMODUS = os.environ.get("DEKADEN_FARBEN", "rot")    # "rot" = Farben wie die Balkengrafiken, "dekade" = Blau bis Rot je Jahrzehnt
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"           # True: dunkles Design
ANZEIGEN = not os.environ.get("CI")                     # auf dem PC die Grafiken auch im Fenster zeigen
START = 1940                                            # erste Dekade
NUR_VOLLE_JAHRE = True                                  # laufendes Jahr nicht in Mittel und Trend
ZEIGE_SPANNE = True                                     # Balken: graue Linie mit der Spanne der fünf Datensätze
ZEIGE_ZUWACHS = True                                    # Balken: Zuwachs zur Vordekade im Balken
GLOCKEN_FLAECHE = 0.12                                  # Glocken: Deckkraft der Füllung (0 = keine)
GLOCKEN_GLAETTUNG = 1.0                                 # Glocken: >1 glatter, <1 detailreicher
PUNKT_GROESSE = 140
TREND_STETIG = True                                     # Punkte: Trendlinien an den Dekadengrenzen verbunden (ohne Sprünge)
PNG_DPI = 150                                           # 20 x 11,25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)
SCHRIFT = 1.3

FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")
SEITE = "https://johnnylauge.github.io/klima-im-wandel/daten/"
CLIMATE_PULSE = angleichung.CLIMATE_PULSE_URL
QS = ["ERA5", "NASA", "NOAA", "HadCRUT5", "Berkeley"]
DREI = ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."]

FARBEN = {
    False: dict(bg="white", fg="black", text="#1d1d1f", box_bg="white", box_rand="lightgray", fuss="gray",
                gitter="#d0d0d4", limit="#b2182b", spanne="#6e6e73"),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50", fuss="#9a9aa2",
                gitter="#4a4a50", limit="#ff6b6b", spanne="#9a9aa2"),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"], "font.weight": "bold",
})
QUELLE_5 = ("Datenquellen: C3S/ECMWF (ERA5, CC BY 4.0) · NASA GISS · NOAA NCEI · HadCRUT5 (Met Office, OGL v3.0) · "
            "Berkeley Earth (CC BY-NC 4.0)\nangeglichen über 1991-2020 · eigene Berechnung · Visualisierung: JohnnyLauge.bsky.social")
QUELLE_ERA5 = ("Datenquelle: Copernicus Climate Pulse (ERA5), C3S/ECMWF · CC BY 4.0\n"
               "https://pulse.climate.copernicus.eu/ · eigene Berechnung\nVisualisierung: JohnnyLauge.bsky.social")


def k(v, n=2):
    return f"{v:+.{n}f}".replace(".", ",").replace("-", "−")


def achse_k(v, _):
    return f"{v:.2f}".replace(".", ",").replace("-", "−")


# =========================================================
# DATEN
# =========================================================
def text_laden(name, url):
    pfad = os.path.join(FOLDER, name)
    if os.path.exists(pfad):
        return open(pfad, encoding="utf-8").read()
    print(f"{name} nicht gefunden - lade {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.text


# Jahreswerte der fünf Datensätze (bereits auf 1850-1900 angeglichen)
vgl = pd.read_csv(io.StringIO(text_laden("VGL_global_jahreswerte.txt", SEITE + "VGL_global_jahreswerte.txt")),
                  comment="#", sep=r"\s+", na_values="-", index_col="Jahr")
vgl = vgl.loc[START:]
if NUR_VOLLE_JAHRE:
    vgl = vgl[vgl["vollständig"] == "ja"]
vgl["dek"] = (vgl.index // 10) * 10

# ERA5-Tageswerte (für die Glockenkurven)
tag = pd.read_csv(io.StringIO(angleichung.climate_pulse_text(text_laden("ERA5_global_tag.txt", CLIMATE_PULSE))), comment="#", parse_dates=["date"])
tag = tag[tag["date"].dt.year >= START].copy()
tag["abw"] = tag["ano_91-20"] + angleichung.offset_tag(tag["date"])
tag["dek"] = (tag["date"].dt.year // 10) * 10
LETZTER = tag["date"].max()

DEK = sorted(vgl["dek"].unique())
mittel = vgl.groupby("dek")["Mittel"].mean()
jahre_je_dek = vgl.groupby("dek").size()


def dek_name(d):
    """'1940er'; eine angefangene Dekade als '2020–25*'."""
    if jahre_je_dek.get(d, 10) < 10:
        return f"{d}–{str(vgl[vgl.dek == d].index.max())[2:]}*"
    return f"{d}er"


# =========================================================
# FARBEN
# =========================================================
def farben_je_dekade():
    if FARBMODUS == "rot":
        lim = float(vgl["Mittel"].abs().max())                 # gleiche Skala wie die Balkengrafiken
        nrm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        return {d: tuple(plt.get_cmap("RdBu_r")(nrm(mittel[d]))[:3]) for d in DEK}
    f = plt.get_cmap("RdYlBu_r")(np.linspace(0, 1, len(DEK)))[:, :3]
    h = rgb_to_hsv(f)
    h[:, 1] = np.clip(h[:, 1] * 1.35 + 0.12, 0, 1)
    if DUNKEL:
        h[:, 2] = np.maximum(h[:, 2], 0.80); h[:, 1] = np.minimum(h[:, 1], 0.85)
    else:
        h[:, 2] = np.minimum(h[:, 2], 0.82)
    return {d: tuple(c) for d, c in zip(DEK, hsv_to_rgb(h))}


FUELL = farben_je_dekade()


def linienfarbe(c):
    """Für Linien und Schrift: blasse Töne auf Weiß abdunkeln, dunkle Töne auf Schwarz aufhellen."""
    if FARBMODUS != "rot":
        return c
    h = rgb_to_hsv(np.array(c))
    if DUNKEL:
        h[2] = max(h[2], 0.85)
    else:
        h[1] = max(h[1], 0.45); h[2] = min(h[2], 0.80)
    return tuple(hsv_to_rgb(h))


LINIE = {d: linienfarbe(c) for d, c in FUELL.items()}


# =========================================================
# GRUNDGERÜST
# =========================================================
def geruest(zeile2, zeile3, quelle):
    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=kopf.DIAGRAMM_OBEN)
    ax.set_title(zeile3, fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mi = ax.annotate(zeile2, xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mi, xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=24 * SCHRIFT, fontweight="bold")
    kopf.setzen(fig, ax)   # Überschrift an festen Stellen (siehe kopf.py)
    fig.text(0.99, 0.01, quelle, ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"], fontweight="normal")
    ax.grid(True, color=FARBEN["gitter"], alpha=0.5, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11 * SCHRIFT)
    return fig, ax


def infokasten(ax, text):
    return ax.text(0.01, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=11 * SCHRIFT,
                   fontweight="bold", linespacing=1.35, zorder=10,
                   bbox=dict(boxstyle="round,pad=0.5", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95))


def legende(ax, unter):
    """Legende als Kasten links unter dem Info-Kasten."""
    ax.figure.canvas.draw()
    b = ax.transAxes.inverted().transform(unter.get_bbox_patch().get_window_extent())
    leg = ax.legend(loc="upper left", bbox_to_anchor=(b[0][0], b[0][1] - 0.012), borderaxespad=0,
                    fontsize=10 * SCHRIFT, facecolor=FARBEN["box_bg"], edgecolor=FARBEN["box_rand"],
                    framealpha=0.95, labelcolor=FARBEN["text"])
    leg.set_zorder(10)
    plt.setp(leg.get_texts(), fontweight="bold")


def speichern(fig, art):
    os.makedirs(PNG_FOLDER, exist_ok=True)
    name = f"Dekaden_{art}" + ("_dekade" if FARBMODUS == "dekade" else "") + ("_dunkel" if DUNKEL else "")
    basis = os.path.join(PNG_FOLDER, name)
    fig.savefig(basis + ".png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    fig.savefig(basis + ".pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    print(f"Grafik gespeichert: {basis}.png")
    if not ANZEIGEN:
        plt.close(fig)


def angefangen_hinweis():
    d = DEK[-1]
    if jahre_je_dek[d] < 10:
        n = jahre_je_dek[d]
        return f"\n* {dek_name(d)[:-1]}: erst {n} Jahr{'e' if n > 1 else ''}"
    return ""


# =========================================================
# 1. BALKEN
# =========================================================
def balken():
    fig, ax = geruest("Global | Dekaden | Mittel aus 5 Datensätzen",
                      "Abweichung von der vorindustriellen Zeit (1850-1900)", QUELLE_5)
    breite = 0.70
    for i, d in enumerate(DEK):
        v = mittel[d]
        teil = vgl[vgl.dek == d]
        ax.bar(i, v, breite, color=FUELL[d], edgecolor=FARBEN["fg"], lw=0.6,
               hatch="//" if jahre_je_dek[d] < 10 else None, zorder=2)
        oben = max(v, 0)
        if ZEIGE_SPANNE:
            werte = [teil[q].mean() for q in QS if teil[q].notna().any()]
            ax.plot([i + breite * 0.38] * 2, [min(werte), max(werte)], color=FARBEN["spanne"], lw=2.5, zorder=3,
                    solid_capstyle="butt")
            oben = max(oben, max(werte))
        ax.text(i, oben + 0.03, k(v) + " K", ha="center", va="bottom", fontsize=13 * SCHRIFT, fontweight="bold", zorder=4)
        if ZEIGE_ZUWACHS and i:
            vor = mittel[DEK[i - 1]]
            dl = v - vor
            schrift = "#1d1d1f" if sum(FUELL[d]) > 1.8 else "white"
            if dl > 0:
                ax.bar(i, dl, breite, bottom=vor, color="white", alpha=0.25, edgecolor="none", zorder=2.5)
                ax.plot([i - breite / 2, i + breite / 2], [vor, vor], color=FARBEN["bg"], lw=2, zorder=2.6)
                ax.text(i, vor + dl / 2 if dl >= 0.07 else vor - 0.035, k(dl), ha="center", va="center",
                        fontsize=12 * SCHRIFT, fontweight="bold", color=schrift, zorder=5)
            else:
                ax.bar(i, -dl, breite, bottom=v, fill=False, edgecolor=FARBEN["spanne"], ls="--", lw=1.4, zorder=2.5)
                ax.text(i, v - 0.035, k(dl), ha="center", va="center", fontsize=12 * SCHRIFT, fontweight="bold",
                        color=schrift, zorder=5)
    ax.axhline(0, color=FARBEN["fg"], lw=1)
    ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=1.8, label="1,5-K-Grenze (Pariser Klimaabkommen)")
    ax.set_xticks(range(len(DEK)), [dek_name(d) for d in DEK], fontsize=14 * SCHRIFT, fontweight="bold")
    ax.set_xlim(-0.6, len(DEK) - 0.4)
    ax.set_ylim(min(0, mittel.min() - 0.05), max(1.75, mittel.max() + 0.3))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(achse_k))
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.grid(axis="x", visible=False)
    zeilen = "Balken: Mittel der Dekade (Durchschnitt von ERA5, NASA, NOAA, HadCRUT5, Berkeley Earth)"
    if ZEIGE_ZUWACHS:
        zeilen += "\nHeller Teil: Zuwachs zur Dekade davor"
    if ZEIGE_SPANNE:
        zeilen += (" · " if ZEIGE_ZUWACHS else "\n") + "graue Linie: Spanne der fünf Datensätze"
    info = infokasten(ax, zeilen + angefangen_hinweis())
    legende(ax, info)
    speichern(fig, "balken")


# =========================================================
# 2. GLOCKENKURVEN (ERA5-Tageswerte)
# =========================================================
def glocken():
    fig, ax = geruest("Global | Dekaden | ERA5",
                      "Verteilung der Tageswerte · Abweichung von der vorindustriellen Zeit (1850-1900)", QUELLE_ERA5)
    xs = np.linspace(tag["abw"].min() - 0.15, tag["abw"].max() + 0.15, 600)
    tdek = sorted(tag["dek"].unique())
    letzte_d = tdek[-1]
    for i, d in enumerate(tdek):
        w = tag.loc[tag["dek"] == d, "abw"].to_numpy()
        bw = 1.06 * w.std() * len(w) ** (-1 / 5) * GLOCKEN_GLAETTUNG                 # Glättung nach Silverman
        y = np.exp(-0.5 * ((xs[:, None] - w[None, :]) / bw) ** 2).sum(1) / (len(w) * bw * np.sqrt(2 * np.pi)) * 100
        f, l = FUELL.get(d, LINIE.get(d)), LINIE.get(d, FUELL.get(d))
        laufend = d == letzte_d and LETZTER.year < d + 9
        if GLOCKEN_FLAECHE:
            ax.fill_between(xs, y, color=f, alpha=GLOCKEN_FLAECHE, lw=0)
        ax.plot(xs, y, color=l, lw=3.2 if laufend else 2.4, ls="--" if laufend else "-")
        name = f"{d}er" + ("*" if laufend else "")
        ax.text(xs[y.argmax()], y.max() + (4 if i % 2 == 0 else 16), name, ha="center", va="bottom",
                fontsize=11 * SCHRIFT, fontweight="bold", color=l)
    ax.axvline(1.5, color=FARBEN["limit"], ls=":", lw=1.8)
    ax.axvline(0, color=FARBEN["fg"], lw=1)
    ax.set_xlabel("Temperaturabweichung eines Tages [K]", fontsize=16 * SCHRIFT, fontweight="bold")
    ax.set_ylabel("Anteil der Tage [% je K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−")))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.set_xlim(xs[0], xs[-1])
    ax.text(1.52, ax.get_ylim()[1] * 0.015, "1,5 K", color=FARBEN["limit"], fontsize=11 * SCHRIFT, va="bottom")
    tm = tag.groupby("dek")["abw"].mean()
    ueber = tag.groupby("dek")["abw"].apply(lambda s: (s > 1.5).mean() * 100)
    zeilen = "\n".join(f"{d}er: Mittel {k(tm[d])} K · {ueber[d]:.0f} % der Tage über 1,5 K".replace(".", ",")
                       for d in tdek[-3:])
    hinweis = (f"\n* {letzte_d}er: erst bis {LETZTER.day}. {DREI[LETZTER.month - 1]} {LETZTER.year}"
               if LETZTER.year < letzte_d + 9 else "")
    infokasten(ax, f"Jede Kurve: wie häufig welche Abweichung in der Dekade vorkam\n{zeilen}{hinweis}")
    speichern(fig, "glocken")


# =========================================================
# 3. PUNKTE je Jahr mit Trend je Dekade
# =========================================================
def punkte():
    fig, ax = geruest("Global | Jahr | Mittel aus 5 Datensätzen",
                      "Abweichung von der vorindustriellen Zeit (1850-1900) · Trend je Dekade", QUELLE_5)
    # stetiger Trend: eine Gerade je Dekade, an den Grenzen verbunden (Regression mit Knickpunkten 1950, 1960 ...)
    if TREND_STETIG:
        x = vgl.index.to_numpy(float)
        knoten = [d for d in DEK[1:]]
        X = np.column_stack([np.ones_like(x), x - DEK[0]] + [np.clip(x - kk, 0, None) for kk in knoten])
        koeff = np.linalg.lstsq(X, vgl["Mittel"].to_numpy(), rcond=None)[0]
        steig = dict(zip(DEK, np.cumsum(koeff[1:])))
        modell = lambda t: (np.column_stack([np.ones_like(t), t - DEK[0]] + [np.clip(t - kk, 0, None) for kk in knoten])
                            @ koeff)
    for d in DEK:
        teil = vgl[vgl.dek == d]
        ax.scatter(teil.index, teil["Mittel"], s=PUNKT_GROESSE, color=FUELL[d], edgecolor=FARBEN["fg"], lw=0.8, zorder=5)
        if TREND_STETIG:
            ende = min(d + 10, float(vgl.index.max()))
            xx = np.array([float(d), ende])
            a = steig[d]
            yy = modell(xx)
            ax.plot(xx, yy, color=FARBEN["bg"], lw=7, zorder=3, solid_capstyle="round")
            ax.plot(xx, yy, color=LINIE[d], lw=4, zorder=4, solid_capstyle="round",
                    ls="--" if jahre_je_dek[d] < 10 else "-")
            ax.text(xx.mean(), teil["Mittel"].max() + 0.12, f"{k(a * 10)} K\npro Dekade", ha="center", va="center",
                    fontsize=10 * SCHRIFT, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=FARBEN["box_bg"], ec=LINIE[d], alpha=0.95))
        elif len(teil) >= 3:
            a, b = np.polyfit(teil.index, teil["Mittel"], 1)
            xx = np.array([teil.index.min(), teil.index.max()])
            ax.plot(xx, a * xx + b, color=FARBEN["bg"], lw=7, zorder=3, solid_capstyle="round")
            ax.plot(xx, a * xx + b, color=LINIE[d], lw=4, zorder=4, solid_capstyle="round",
                    ls="--" if jahre_je_dek[d] < 10 else "-")
            ax.text(xx.mean(), teil["Mittel"].max() + 0.12, f"{k(a * 10)} K\npro Dekade", ha="center", va="center",
                    fontsize=10 * SCHRIFT, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=FARBEN["box_bg"], ec=LINIE[d], alpha=0.95))
    ax.axhline(0, color=FARBEN["fg"], lw=1)
    ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=1.8, label="1,5-K-Grenze (Pariser Klimaabkommen)")
    ax.set_xlim(START - 1.5, vgl.index.max() + 1.5)
    ax.set_xticks(range(START, vgl.index.max() + 1, 10))
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=18 * SCHRIFT, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(achse_k))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.05, hi + 0.30 * (hi - lo))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    g = np.polyfit(vgl.index, vgl["Mittel"], 1)[0] * 10
    info = infokasten(ax, "Punkte: Jahresmittel (Durchschnitt aus fünf Datensätzen) · Linien: linearer Trend je Dekade"
                          + (", an den Grenzen verbunden" if TREND_STETIG else "") + "\n"
                          f"Zum Vergleich: Trend {START} bis {vgl.index.max()} {k(g)} K pro Dekade")
    legende(ax, info)
    speichern(fig, "punkte")


if __name__ == "__main__":
    for art in WAHL:
        {"balken": balken, "glocken": glocken, "punkte": punkte}[art.strip()]()
    if ANZEIGEN:
        plt.show()
