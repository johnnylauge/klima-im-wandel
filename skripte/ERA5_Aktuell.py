"""
Abschnitt "Aktuell" der Webseite: globale Temperatur (ERA5, Copernicus Climate Pulse) des laufenden
Jahres im Vergleich zu allen Jahren seit 1940 - vier Fassungen:

  tag_c    Tageswerte in °C           tag_k    Tageswerte als Abweichung von 1850-1900 (mit 1,5-K-Linie)
  monat_c  Monatswerte in °C          monat_k  Monatswerte als Abweichung von 1850-1900

Jede Linie ist ein Jahr, eingefärbt nach Jahrzehnt (1940er blau ... 2020er rot, Farbleiste oben rechts).
Das laufende Jahr ist weiß (dunkel) bzw. schwarz (hell) hervorgehoben, dazu die Tagesrekorde bzw.
Monatsrekorde (gestrichelt) und ein Info-Kasten oben links.

Dateien (in ERA5_FOLDER/Grafiken):
  ERA5_aktuell_<fassung>[_dunkel]_voll.png/.pdf   mit Info-Kasten (Vergrößern, Download)
  ERA5_aktuell_<fassung>[_dunkel].png             ohne Info-Kasten - die Webseite zeigt die Werte
  ERA5_aktuell_<fassung>.json                     Lage und Werte jedes Tages bzw. Monats (Maus)
und ERA5_FOLDER/ERA5_aktuell.json mit den Werten für die Kacheln.

Umgebungsvariablen: ERA5_FOLDER, ERA5_OHNE_DOWNLOAD=1, ERA5_DUNKEL=1, AKTUELL_WAHL (z. B. "tag_k",
Standard: alle vier). Lokal (ohne CI) wird nur die volle Fassung gespeichert.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("CI"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D


# Achsenbeschriftung in deutscher Schreibweise (1,5 statt 1.5) - gilt für alle Achsen ohne eigene Formatierung
from matplotlib.ticker import ScalarFormatter as _SF
if not getattr(_SF, "_komma", False):
    _sf_alt = _SF.__call__
    _SF.__call__ = lambda self, x, pos=None: _sf_alt(self, x, pos).replace(".", ",")
    _SF._komma = True

# =========================================================
# EINSTELLUNGEN
# =========================================================
PNG_DPI = 150                   # 20 x 11.25 Zoll x 150 dpi = 3000 x 1688 Pixel
GROESSE = (20, 11.25)
SCHRIFT = 1.3
START = 1940
DUNKEL = os.environ.get("ERA5_DUNKEL") == "1"
WEB = os.environ.get("ERA5_WEB", "1" if os.environ.get("CI") else "0") == "1"
FASSUNGEN = ["tag_c", "tag_k", "monat_c", "monat_k"]
if os.environ.get("AKTUELL_WAHL"):
    FASSUNGEN = [f.strip() for f in os.environ["AKTUELL_WAHL"].split(",")]

# Vorindustrielle Offsets je Monat (wie in den anderen ERA5-Skripten)
MONTHLY_OFFSETS = dict(zip(range(1, 13), [0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93]))
MONATE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
MONATSNAMEN = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
               "Oktober", "November", "Dezember"]
DREI = ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."]

FARBEN = {
    False: dict(bg="white", fg="black", text="#1d1d1f", box_bg="white", box_rand="lightgray", fuss="gray",
                jahr="#111111", jahr_rand="white", rekord="#3a3a40", limit="#b2182b", gitter="#d0d0d4",
                alpha_tag=0.55, alpha_monat=0.85),
    True:  dict(bg="#1e1e21", fg="#e6e6e9", text="#ececee", box_bg="#2a2a2e", box_rand="#4a4a50", fuss="#9a9aa2",
                jahr="#ffffff", jahr_rand="#1e1e21", rekord="#ececee", limit="#ff6b6b", gitter="#4a4a50",
                alpha_tag=0.35, alpha_monat=0.8),
}[DUNKEL]
plt.rcParams.update({
    "text.color": FARBEN["text"], "axes.labelcolor": FARBEN["text"],
    "xtick.color": FARBEN["text"], "ytick.color": FARBEN["text"],
    "axes.edgecolor": FARBEN["fg"], "axes.facecolor": FARBEN["bg"],
    "figure.facecolor": FARBEN["bg"],
})

URL = "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv"
FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
FILE = os.path.join(FOLDER, "ERA5_global_tag.txt")
PNG_FOLDER = os.path.join(FOLDER, "Grafiken")
QUELLE = ("Datenquelle: Copernicus Climate Pulse (ERA5), C3S/ECMWF · CC BY 4.0\n"
          "https://pulse.climate.copernicus.eu/ · eigene Berechnung\n"
          "Visualisierung: JohnnyLauge.bsky.social")


def tag_monat(d):
    return f"{d.day}. {DREI[d.month - 1]}"


def komma(v, plus=False):
    return (f"{v:+.2f}" if plus else f"{v:.2f}").replace(".", ",").replace("-", "−")


# =========================================================
# DATEN
# =========================================================
if not (os.environ.get("ERA5_OHNE_DOWNLOAD") == "1" and os.path.exists(FILE)):
    print("Lade aktuelle ERA5-Daten herunter...")
    os.makedirs(FOLDER, exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with open(FILE, "wb") as f:
        f.write(r.content)

df = pd.read_csv(FILE, comment="#", parse_dates=["date"])
df = df[df["date"].dt.year >= START].copy()
df["jahr"] = df["date"].dt.year
df["monat"] = df["date"].dt.month
df["abw"] = df["ano_91-20"] + df["monat"].map(MONTHLY_OFFSETS)
# feste Position im Jahr (1-365), der 29. Februar fällt heraus
df["pos"] = pd.to_datetime("2001-" + df["date"].dt.strftime("%m-%d"), errors="coerce").dt.dayofyear
alle_tage = df.copy()                       # mit 29. Februar - zum Zählen der Tage über einer Schwelle
df = df.dropna(subset=["pos"]).astype({"pos": int})

LETZTER = df["date"].max()
JAHR = int(LETZTER.year)
VORJAHR = JAHR - 1
akt = df[df["jahr"] == JAHR]
monat_voll = LETZTER == LETZTER + pd.offsets.MonthEnd(0)     # letzter Monat schon vollständig?

# Monatsmittel je Jahr (der laufende Monat nur mit den vorhandenen Tagen)
monate = df.groupby(["jahr", "monat"]).agg(c=("2t", "mean"), k=("abw", "mean"), n=("2t", "size")).reset_index()

# Jahreswerte für Rang (vollständige Jahre plus das laufende bisher)
jahre = df.groupby("jahr").agg(k=("abw", "mean"), n=("abw", "size"))
jahre = jahre[(jahre["n"] >= 365) | (jahre.index == JAHR)]
rang = int((jahre["k"] > jahre.loc[JAHR, "k"]).sum()) + 1

# Farben je Jahrzehnt
DEKADEN = list(range(START, (JAHR // 10) * 10 + 10, 10))
# Kräftigere Farben als die Standard-Palette: hell nichts zu Helles (Gelb und Hellblau verschwinden sonst
# auf Weiß), dunkel nichts zu Dunkles (das tiefe Blau der 1940er verschwindet sonst im Hintergrund).
def _palette(n):
    from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
    farben = plt.get_cmap("RdYlBu_r")(np.linspace(0, 1, n))[:, :3]
    hsv = rgb_to_hsv(farben)
    hsv[:, 1] = np.clip(hsv[:, 1] * 1.35 + 0.12, 0, 1)              # mehr Farbe (hell und dunkel)
    if DUNKEL:
        hsv[:, 2] = np.maximum(hsv[:, 2], 0.80)
        hsv[:, 1] = np.minimum(hsv[:, 1], 0.85)                    # und nicht ganz so grell                    # tiefes Blau nicht im dunklen Hintergrund versinken lassen
    else:
        hsv[:, 2] = np.minimum(hsv[:, 2], 0.82)                    # nichts heller als 82 %
    farben = hsv_to_rgb(hsv)
    return ListedColormap(farben)


CMAP = _palette(len(DEKADEN))
NORM = BoundaryNorm(DEKADEN + [DEKADEN[-1] + 10], len(DEKADEN))

# =========================================================
# KACHELN (eine Datei für die Webseite)
# =========================================================
tag_tab = df.pivot(index="pos", columns="jahr", values="2t")
tag_abw = df.pivot(index="pos", columns="jahr", values="abw")
alt = [j for j in tag_tab.columns if j < JAHR]
letzte = akt.iloc[-1]
rekord_c = tag_tab.loc[letzte["pos"], alt].max()
rekord_j = int(tag_tab.loc[letzte["pos"], alt].idxmax())
# Rang des neuesten Tages unter allen gleichen Kalendertagen seit 1940
heute = tag_abw.loc[letzte["pos"]].dropna()
rang_tag = int((heute > letzte["abw"]).sum()) + 1
rekord_abw = float(tag_abw.loc[letzte["pos"], alt].max())

# laufender Monat: Rang unter allen gleichen Monaten, Rekord der Vorjahre
m_jetzt = monate[monate["jahr"] == JAHR].iloc[-1]
gleich = monate[monate["monat"] == m_jetzt["monat"]]
m_vorher = gleich[gleich["jahr"] < JAHR]
m_rek = m_vorher.loc[m_vorher["k"].idxmax()]


def schwelle(s):
    """Tage und Monate über s K: laufendes Jahr, Rekord bis zum gleichen Tag/Monat, Rekordjahre, erstmals."""
    def bester(reihe):
        top = reihe.max()
        jahre = [int(j) for j in reihe[reihe == top].index[::-1]]              # neuestes zuerst
        return dict(wert=int(top), jahre=jahre[:3], anzahl=len(jahre))
    def rangliste(reihe):                                                       # gleich viele: neueres Jahr zuerst
        t = reihe.reset_index().sort_values([reihe.name, "jahr"], ascending=[False, False]).head(3)
        return [[int(j), int(v)] for j, v in zip(t["jahr"], t[reihe.name])]
    akt_alle = alle_tage[alle_tage["jahr"] == JAHR]
    vor = alle_tage[alle_tage["jahr"] < JAHR]
    bis_heute = vor[(vor["monat"] < LETZTER.month) | ((vor["monat"] == LETZTER.month) & (vor["date"].dt.day <= LETZTER.day))].groupby("jahr")["abw"].apply(lambda x: int((x > s).sum()))
    voll_t = vor.groupby("jahr")["abw"].apply(lambda x: int((x > s).sum()))
    mv = monate[monate["jahr"] < JAHR]
    n_mon = int((monate["jahr"] == JAHR).sum())
    bis_mon = mv[mv["monat"] <= n_mon].groupby("jahr")["k"].apply(lambda x: int((x > s).sum()))
    voll_m = mv.groupby("jahr")["k"].apply(lambda x: int((x > s).sum()))
    erst_t = alle_tage.loc[alle_tage["abw"] > s, "date"].min()
    erst_m = monate[monate["k"] > s].sort_values(["jahr", "monat"]).head(1)
    return dict(
        schwelle=s,
        tage=int((akt_alle["abw"] > s).sum()), von=int(len(akt_alle)),
        tage_rekord=bester(bis_heute), tage_voll=rangliste(voll_t),
        tage_erstmals=f"{erst_t:%d.%m.%Y}" if pd.notna(erst_t) else None,
        monate=int((monate[monate["jahr"] == JAHR]["k"] > s).sum()), von_monate=n_mon,
        monate_rekord=bester(bis_mon), monate_voll=rangliste(voll_m),
        monate_erstmals=(f"{MONATSNAMEN[int(erst_m.iloc[0]['monat']) - 1]} {int(erst_m.iloc[0]['jahr'])}"
                         if len(erst_m) else None))


kacheln = dict(
    stand=f"{LETZTER:%d.%m.%Y}", jahr=JAHR,
    tag=dict(datum=f"{LETZTER:%d.%m.%Y}", temp=round(float(letzte["2t"]), 3), abw=round(float(letzte["abw"]), 3),
             rang=rang_tag, von=int(len(heute)),
             rekord=round(float(rekord_c), 3), rekord_abw=round(rekord_abw, 3), rekord_jahr=rekord_j,
             abstand=round(float(letzte["2t"] - rekord_c), 3)),
    monat=dict(monat=int(m_jetzt["monat"]), name=MONATSNAMEN[int(m_jetzt["monat"]) - 1], voll=bool(monat_voll),
               temp=round(float(m_jetzt["c"]), 3), abw=round(float(m_jetzt["k"]), 3),
               rang=int((gleich["k"] > m_jetzt["k"]).sum()) + 1, von=int(len(gleich)),
               rekord=round(float(m_rek["c"]), 3), rekord_abw=round(float(m_rek["k"]), 3), rekord_jahr=int(m_rek["jahr"])),
    schwellen=[schwelle(s) for s in (1.0, 1.5, 2.0)],
    ueber15=dict(tage=int((akt["abw"] > 1.5).sum()), von=int(len(akt)),
                 monate=int((monate[(monate["jahr"] == JAHR)]["k"] > 1.5).sum()),
                 von_monate=int((monate["jahr"] == JAHR).sum())),
    jahr_bisher=dict(temp=round(float(akt["2t"].mean()), 3), abw=round(float(akt["abw"].mean()), 3),
                     rang=rang, von=int(len(jahre))),
)
with open(os.path.join(FOLDER, "ERA5_aktuell.json"), "w", encoding="utf-8") as f:
    json.dump(kacheln, f, ensure_ascii=False)


# =========================================================
# GRAFIK
# =========================================================
def zeichne(fassung):
    art, einheit = fassung.split("_")          # "tag"/"monat", "c"/"k"
    spalte = "2t" if einheit == "c" else "abw"
    fig, ax = plt.subplots(figsize=GROESSE, facecolor=FARBEN["bg"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.87)

    if art == "tag":
        tab = tag_tab if einheit == "c" else tag_abw
        x = tab.index.to_numpy()
        breite_linie, alpha = 0.8, FARBEN["alpha_tag"]
    else:
        tab = monate.pivot(index="monat", columns="jahr", values="c" if einheit == "c" else "k")
        x = tab.index.to_numpy()
        breite_linie, alpha = 1.6, FARBEN["alpha_monat"]
    vorher = [j for j in tab.columns if j < JAHR]

    # alle früheren Jahre, eingefärbt nach Jahrzehnt
    for j in vorher:
        ax.plot(x, tab[j].to_numpy(), color=CMAP(NORM(j)), lw=breite_linie, alpha=alpha, zorder=1)

    # Rekorde (gestrichelt)
    warm, kalt = tab[vorher].max(axis=1), tab[vorher].min(axis=1)
    wort = "tägl." if art == "tag" else "monatl."
    ax.plot(x, warm.to_numpy(), color=FARBEN["rekord"], lw=1.4 if art == "tag" else 1.8, ls=(0, (5, 3)), alpha=0.9,
            zorder=2.5, label=f"{wort} Wärmerekord {START}-{VORJAHR}")
    ax.plot(x, kalt.to_numpy(), color=FARBEN["rekord"], lw=1.4 if art == "tag" else 1.8, ls=(0, (1.5, 2.5)), alpha=0.9,
            zorder=2.5, label=f"{wort} Kälterekord {START}-{VORJAHR}")

    # 1,5-K-Linie bei der Abweichung
    if einheit == "k":
        ax.axhline(1.5, color=FARBEN["limit"], ls=":", lw=2.2, zorder=2, label="1,5-K-Grenze (Pariser Klimaabkommen)")
        ax.axhline(0, color=FARBEN["fg"], lw=0.9, zorder=1.5)

    # laufendes Jahr
    cur = tab[JAHR].dropna()
    if art == "tag":
        ax.plot(cur.index, cur.values, color=FARBEN["jahr_rand"], lw=6, zorder=3, solid_capstyle="round")
        ax.plot(cur.index, cur.values, color=FARBEN["jahr"], lw=3, zorder=4, label=f"{JAHR}")
        ax.scatter([cur.index[-1]], [cur.values[-1]], s=80, color=FARBEN["jahr"], edgecolor=FARBEN["jahr_rand"],
                   lw=1.5, zorder=5)
    else:
        # ohne Punkte; der angefangene Monat gestrichelt (mit Rand, damit er zwischen den Linien sichtbar bleibt)
        import matplotlib.patheffects as pe
        voll = cur if monat_voll else cur.iloc[:-1]
        ax.plot(voll.index, voll.values, color=FARBEN["jahr_rand"], lw=7, zorder=3, solid_capstyle="round")
        ax.plot(voll.index, voll.values, color=FARBEN["jahr"], lw=4, zorder=4, solid_capstyle="round", label=f"{JAHR}")
        if not monat_voll and len(cur) >= 2:
            teil = cur.iloc[-2:]
            ax.plot(teil.index, teil.values, color=FARBEN["jahr"], lw=4, ls=(0, (2.5, 1.5)), zorder=4,
                    dash_capstyle="butt", label=f"{MONATSNAMEN[LETZTER.month - 1]} (bisher)",
                    path_effects=[pe.Stroke(linewidth=7, foreground=FARBEN["jahr_rand"]), pe.Normal()])

    # Achsen
    if art == "tag":
        starts = pd.date_range("2001-01-01", periods=12, freq="MS").dayofyear
        ax.set_xticks(starts, MONATE)
        ax.set_xlim(1, 365)
    else:
        ax.set_xticks(range(1, 13), MONATE)
        ax.set_xlim(1, 12)                                           # Linien beginnen und enden an der Achse
    werte = pd.concat([tab[j] for j in tab.columns]).dropna()
    unten, oben = float(werte.min()), float(werte.max())
    if einheit == "k":
        oben = max(oben, 1.6)
    spanne = oben - unten
    ax.set_ylim(unten - 0.04 * spanne, oben + (0.34 if einheit == "k" else 0.20) * spanne)   # oben Platz für Info-Kasten (und Legende)
    # Abweichung mit zwei Nachkommastellen wie bei den anderen Grafiken (1,00 · 1,50 ...)
    fmt = (lambda v, _: f"{v:.0f}" if einheit == "c" and abs(v - round(v)) < 1e-9
           else f"{v:.2f}".replace(".", ",").replace("-", "−"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
    ax.set_ylabel("Temperatur [°C]" if einheit == "c" else "Temperaturabweichung [K]",
                  fontsize=18 * SCHRIFT, fontweight="bold")
    ax.tick_params(axis="both", labelsize=11 * SCHRIFT)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    # Achsentitel genauso weit außen wie bei den anderen Grafiken (deren breiteste Beschriftung ist z. B. "−0,25")
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    lo_, hi_ = ax.get_ylim()
    breit = max((t.get_window_extent(rr).width for t, v in zip(ax.get_yticklabels(), ax.get_yticks())
                 if t.get_text() and lo_ <= v <= hi_), default=0)                     # nur sichtbare Beschriftungen
    probe = fig.text(0, 0, "−0,25", fontsize=11 * SCHRIFT, fontweight="bold")
    ziel = probe.get_window_extent(rr).width
    probe.remove()
    ax.yaxis.labelpad = plt.rcParams["axes.labelpad"] + max(0, ziel - breit) * 72 / fig.dpi
    ax.grid(True, color=FARBEN["gitter"], alpha=0.5, lw=0.8)
    ax.set_axisbelow(True)

    # Überschrift in drei Zeilen wie bei den anderen Grafiken
    # °C: was gezeigt wird · Abweichung: gleiche dritte Zeile wie bei den anderen Grafiken
    zeile3 = ("Abweichung von der vorindustriellen Zeit (1850-1900)" if einheit == "k"
              else ("Tageswerte" if art == "tag" else "Monatswerte") + f" im Vergleich zu den Jahren {START}-{VORJAHR}")
    ax.set_title(zeile3, fontsize=15 * SCHRIFT, fontweight="normal", pad=8)
    mitte = ax.annotate(f"Global | {JAHR} | ERA5", xy=(0.5, 1), xycoords=ax.title, xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=16 * SCHRIFT, fontweight="bold")
    ax.annotate("Temperaturmittel", xy=(0.5, 1), xycoords=mitte, xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=24 * SCHRIFT, fontweight="bold")
    fig.text(0.99, 0.01, QUELLE, ha="right", va="bottom", fontsize=9 * SCHRIFT, color=FARBEN["fuss"])

    # Farbleiste der Jahrzehnte: waagrecht in einem Kasten oben rechts (gleicher Stil wie der Info-Kasten),
    # beschriftet nur an den Enden ("1940er" ... "2020er")
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.765, 0.885), 0.226, 0.09, boxstyle="round,pad=0.004,rounding_size=0.008",
                                transform=ax.transAxes, fc=FARBEN["box_bg"], ec=FARBEN["box_rand"],
                                alpha=0.95, zorder=9))
    cax = ax.inset_axes([0.775, 0.905, 0.206, 0.03])
    cax.set_zorder(11)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[DEKADEN[0] + 5, DEKADEN[-1] + 5])
    cb.ax.set_xticklabels([f"{DEKADEN[0]}er", f"{DEKADEN[-1]}er"], fontsize=10 * SCHRIFT, fontweight="bold")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(length=0, pad=3)
    cb.ax.get_xticklabels()[0].set_ha("left")
    cb.ax.get_xticklabels()[-1].set_ha("right")
    # Beschriftung genau an die Enden der Leiste schieben
    from matplotlib.transforms import ScaledTranslation
    cb.ax.get_xticklabels()[0].set_transform(cb.ax.get_xticklabels()[0].get_transform()
        + ScaledTranslation(-cax.get_window_extent().width / len(DEKADEN) / 2 / fig.dpi, 0, fig.dpi_scale_trans))
    cb.ax.get_xticklabels()[-1].set_transform(cb.ax.get_xticklabels()[-1].get_transform()
        + ScaledTranslation(cax.get_window_extent().width / len(DEKADEN) / 2 / fig.dpi, 0, fig.dpi_scale_trans))
    cb.outline.set_edgecolor(FARBEN["box_rand"])

    # Info-Kasten oben links
    bis = f" (bis {tag_monat(LETZTER)})"
    # deutsche Schreibweise mit Komma ("15,07 °C" - mit Punkt sähe es aus wie ein Datum)
    c_, k_ = (lambda v: komma(v)), (lambda v: komma(v, True))
    if art == "tag":
        z1 = (f"{tag_monat(LETZTER)} {JAHR}: {c_(letzte['2t'])} °C | {k_(letzte['abw'])} K"
              f"  (Tagesrekord {c_(rekord_c)} °C, {rekord_j})")
        z2 = (f"Tage über 1,5 K in {JAHR}: {kacheln['ueber15']['tage']} von {kacheln['ueber15']['von']}" if einheit == "k"
              else f"Temperaturmittel {JAHR}{bis}: {c_(akt['2t'].mean())} °C | {k_(akt['abw'].mean())} K")
    else:
        m = monate[monate["jahr"] == JAHR].iloc[-1]
        z1 = (f"{MONATSNAMEN[int(m['monat']) - 1]} {JAHR}{'' if monat_voll else ' (bisher)'}: "
              f"{c_(m['c'])} °C | {k_(m['k'])} K")
        z2 = (f"Monate über 1,5 K in {JAHR}: {kacheln['ueber15']['monate']} von {kacheln['ueber15']['von_monate']}"
              if einheit == "k" else f"Temperaturmittel {JAHR}{bis}: {c_(akt['2t'].mean())} °C | {k_(akt['abw'].mean())} K")
    info = ax.text(0.01, 0.97, f"{z1}\n{z2}", transform=ax.transAxes, ha="left", va="top",
                   fontsize=11 * SCHRIFT, fontweight="bold", linespacing=1.35, zorder=10,
                   bbox=dict(boxstyle="round,pad=0.5", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95))

    # Legende als Kasten im Diagramm (gleicher Stil wie der Info-Kasten):
    stil = dict(fontsize=10 * SCHRIFT, facecolor=FARBEN["box_bg"], edgecolor=FARBEN["box_rand"], framealpha=0.95,
                labelcolor=FARBEN["text"], handlelength=2.4, columnspacing=1.4, borderpad=0.6)
    # unter dem Info-Kasten links; bei der Abweichung in zwei Zeilen (sonst läge sie in den Linien)
    fig.canvas.draw()
    kasten = ax.transAxes.inverted().transform(info.get_bbox_patch().get_window_extent())
    n = len(ax.get_legend_handles_labels()[0])
    leg = ax.legend(loc="upper left", bbox_to_anchor=(kasten[0][0], kasten[0][1] - 0.012),
                    ncol=1 if einheit == "c" else (n + 1) // 2, borderaxespad=0, **stil)
    leg.set_zorder(10)
    leg.get_frame().set_boxstyle("round,pad=0.2,rounding_size=0.6")
    plt.setp(leg.get_texts(), fontweight="bold")

    # Kasten am neuesten Wert (nur in PNG/PDF zum Herunterladen - auf der Webseite zeigt das die Maus-Anzeige)
    letzt = tab[JAHR].dropna()
    if len(letzt):
        x_l, y_l = letzt.index[-1], letzt.iloc[-1]
        if art == "tag":
            titel = f"{tag_monat(LETZTER)} {JAHR}"
            c_w, k_w = letzte["2t"], letzte["abw"]
        else:
            mm = monate[monate["jahr"] == JAHR].iloc[-1]
            titel = f"{MONATSNAMEN[int(mm['monat']) - 1]} {JAHR}" + ("" if monat_voll else " (bisher)")
            c_w, k_w = mm["c"], mm["k"]
        # Kasten über den Linien, mit waagrechter und senkrechter Linie zum Punkt (wie beim wärmsten Jahr),
        # damit er keine Daten verdeckt. Gesucht wird der erste freie Platz von unten nach oben.
        from matplotlib.patches import FancyBboxPatch as _FB
        huelle = tab.max(axis=1)                                     # höchster Wert aller Jahre je Tag/Monat
        unten_y, oben_y = ax.get_ylim()
        hoehe = oben_y - unten_y
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        hindernisse = [p for p in ax.patches if isinstance(p, _FB)] + [info.get_bbox_patch()]
        if ax.get_legend():
            hindernisse.append(ax.get_legend().get_frame())
        inv = ax.transData.inverted()

        def kasten(links, y):
            dx = (20 if art == "tag" else 0.35) * (-1 if links else 1)
            return ax.annotate(
                f"{titel}\n{c_(c_w)} °C | {k_(k_w)} K", xy=(x_l, y_l), xytext=(x_l + dx, y),
                ha="right" if links else "left", va="bottom", multialignment="right" if links else "left",
                fontsize=11 * SCHRIFT, fontweight="bold", zorder=11, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=FARBEN["fuss"], lw=1.2,
                                connectionstyle="angle,angleA=0,angleB=90,rad=0",
                                relpos=(1, 0.5) if links else (0, 0.5), shrinkA=0, shrinkB=4))

        def kasten_oben(y):
            """Kasten direkt über dem Punkt, nur mit senkrechtem Strich; am Rand so weit eingerückt, dass er in die Grafik passt."""
            m = ax.annotate(f"{titel}\n{c_(c_w)} °C | {k_(k_w)} K", xy=(x_l, y_l), xytext=(x_l, y),
                            ha="center", va="bottom", multialignment="center",
                            fontsize=11 * SCHRIFT, fontweight="bold", zorder=11, linespacing=1.3,
                            bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95))
            m.update_positions(r)
            m.update_bbox_position_size(r)
            bb = m.get_bbox_patch().get_window_extent(r)
            achse = ax.get_window_extent(r)
            # so weit nach links/rechts einrücken, dass der Kasten ganz in der Grafik bleibt
            p_px = ax.transData.transform((x_l, y))[0]
            verschieben = max(0, achse.x0 + 6 - bb.x0) - max(0, bb.x1 - (achse.x1 - 6))
            mitte_px = p_px + verschieben
            x_neu = ax.transData.inverted().transform((mitte_px, 0))[0]
            m.remove()
            anteil = 0.5 - verschieben / bb.width                       # Stelle am unteren Rand genau über dem Punkt
            return ax.annotate(f"{titel}\n{c_(c_w)} °C | {k_(k_w)} K", xy=(x_l, y_l), xytext=(x_neu, y),
                               ha="center", va="bottom", multialignment="center",
                               fontsize=11 * SCHRIFT, fontweight="bold", zorder=11, linespacing=1.3,
                               bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95),
                               arrowprops=dict(arrowstyle="-", color=FARBEN["fuss"], lw=1.2,
                                               relpos=(min(max(anteil, 0.04), 0.96), 0), shrinkA=0, shrinkB=4),
                               gid="oben")

        def frei(m):
            m.update_positions(r)                                     # nur den Kasten berechnen (schneller als alles zeichnen)
            m.update_bbox_position_size(r)
            bb = m.get_bbox_patch().get_window_extent(r)
            achse = ax.get_window_extent(r)
            if bb.x0 < achse.x0 + 2 or bb.x1 > achse.x1 - 2 or bb.y1 > achse.y1 - 2:
                return False
            if any(bb.overlaps(h.get_window_extent(r)) for h in hindernisse):
                return False
            (x0, y0), (x1, _) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            # Hülle auch zwischen den Monaten (die Linien verbinden die Monatswerte gerade)
            rand = 3 if art == "tag" else 0.15
            xs = np.linspace(x0 - rand, x1 + rand, 200)
            spitze = np.interp(xs, huelle.index.to_numpy(float), huelle.to_numpy(float))
            return y0 > spitze.max() + 0.02 * hoehe

        # erste Jahreshälfte: Kasten nach links (dort liegen die Linien tiefer), zweite Hälfte: nach rechts
        seiten = (True, False) if x_l <= (182 if art == "tag" else 6.5) else (False, True)
        # ganz am Anfang oder Ende des Jahres zuerst direkt darüber (nur senkrechter Strich), sonst seitlich
        am_rand = (x_l < 40 or x_l > 325) if art == "tag" else (x_l <= 1.5 or x_l >= 11.5)
        seiten = (("oben",) + seiten) if am_rand else (seiten + ("oben",))
        punkt_kasten = None
        start = float(y_l)                                           # Suche beginnt auf Höhe des Punkts -> kurze Linie

        for _ in range(8):                                           # notfalls oben etwas mehr Platz
            for schritt in range(0, 80):
                y = start + (0.02 + 0.01 * schritt) * hoehe
                for links in seiten:
                    m = kasten_oben(y) if links == "oben" else kasten(links, y)
                    if frei(m):
                        punkt_kasten = m
                        break
                    m.remove()
                if punkt_kasten is not None or y > oben_y:
                    break
            if punkt_kasten is not None:
                break
            ax.set_ylim(unten_y, oben_y + 0.06 * hoehe)
            unten_y, oben_y = ax.get_ylim()
        if punkt_kasten is None:                                     # Notlösung: direkt neben den Punkt
            punkt_kasten = ax.annotate(
                f"{titel}\n{c_(c_w)} °C | {k_(k_w)} K", xy=(x_l, y_l), xytext=(-14, 0), textcoords="offset points",
                ha="right", va="center", multialignment="right", fontsize=11 * SCHRIFT, fontweight="bold", zorder=11,
                bbox=dict(boxstyle="round,pad=0.35", fc=FARBEN["box_bg"], ec=FARBEN["box_rand"], alpha=0.95))
    else:
        punkt_kasten = None

    # Speichern
    os.makedirs(PNG_FOLDER, exist_ok=True)
    name = f"ERA5_aktuell_{fassung}"
    basis = os.path.join(PNG_FOLDER, name + ("_dunkel" if DUNKEL else ""))
    fig.savefig(basis + "_voll.png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    fig.savefig(basis + "_voll.pdf", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    if WEB:
        speichere_hover(fig, ax, info, art, einheit, tab, warm, kalt, os.path.join(PNG_FOLDER, name + ".json"))
        info.set_visible(False)
        if punkt_kasten is not None:
            punkt_kasten.set_visible(False)
        fig.savefig(basis + ".png", dpi=PNG_DPI, facecolor=FARBEN["bg"])
    print(f"Grafik gespeichert: {basis}_voll.png")
    plt.close(fig)


def speichere_hover(fig, ax, info, art, einheit, tab, warm, kalt, datei):
    """Je Tag bzw. Monat ein Feld über die ganze Höhe des Diagramms; 'text' steht fertig im Kasten."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    breite, hoehe = fig.canvas.get_width_height()
    fx = lambda v: round(min(max(v / breite, 0), 1), 5)
    fy = lambda v: round(min(max(1 - v / hoehe, 0), 1), 5)
    zu_px = ax.transData.transform
    vorher = [j for j in tab.columns if j < JAHR]
    w_jahr, k_jahr = tab[vorher].idxmax(axis=1), tab[vorher].idxmin(axis=1)
    e = " °C" if einheit == "c" else " K"
    zahl = (lambda v: komma(v)) if einheit == "c" else (lambda v: komma(v, True))
    felder = []
    for x in tab.index:
        if art == "tag":
            datum = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(x) - 1)
            wann = tag_monat(datum)
        else:
            wann = MONATSNAMEN[int(x) - 1]
        rek = f"Wärmerekord {zahl(warm[x])}{e} ({w_jahr[x]}) | Kälterekord {zahl(kalt[x])}{e} ({k_jahr[x]})"
        wert = tab.loc[x, JAHR] if JAHR in tab.columns else np.nan
        if pd.notna(wert):
            z = (akt[akt["pos"] == x].iloc[0] if art == "tag"
                 else monate[(monate["jahr"] == JAHR) & (monate["monat"] == x)].iloc[0])
            c, kk = (z["2t"], z["abw"]) if art == "tag" else (z["c"], z["k"])
            bisher = " (bisher)" if art == "monat" and int(x) == LETZTER.month and not monat_voll else ""
            # "neuer Rekord" nicht für einen erst angefangenen Monat
            neu = " · <strong>neuer Rekord</strong>" if wert > warm[x] and not bisher else ""
            text = f"{wann} {JAHR}{bisher}: {komma(c)} °C | {komma(kk, True)} K{neu}<br>{rek}"
        else:
            text = f"{wann}: noch keine Daten für {JAHR}<br>{rek}"
        halb = 0.5
        links_x, rechts_x = ax.get_xlim()                            # Felder am Rand nicht über die Achse hinaus
        x0, x1 = zu_px((max(x - halb, links_x), 0))[0], zu_px((min(x + halb, rechts_x), 0))[0]
        y0, y1 = zu_px((x, warm[x]))[1], zu_px((x, kalt[x]))[1]
        felder.append(dict(text=text, x0=fx(x0), x1=fx(x1), y0=fy(y0), y1=fy(y1)))
    werte_jahr = tab[JAHR].dropna() if JAHR in tab.columns else pd.Series(dtype=float)
    neuester = int(list(tab.index).index(werte_jahr.index[-1])) if len(werte_jahr) else len(felder) - 1
    k = info.get_bbox_patch().get_window_extent(r)
    a = ax.get_window_extent(r)
    with open(datei, "w", encoding="utf-8") as fh:
        json.dump(dict(achse=dict(x0=fx(a.x0), x1=fx(a.x1), y0=fy(a.y1), y1=fy(a.y0)),
                       info=dict(x0=fx(k.x0), x1=fx(k.x1), y0=fy(k.y1), y1=fy(k.y0)),
                       neuester=neuester, balken=felder), fh, ensure_ascii=False)


for fassung in FASSUNGEN:
    zeichne(fassung)

if not os.environ.get("CI") and not WEB:
    plt.show()
