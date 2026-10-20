"""
Niño 3.4 – tägliche Abweichung der Meeresoberflächentemperatur als Jahresvergleich (Klima-im-Wandel-Stil)

Daten: NOAA OISST v2.1 (Variable 'anom', Bezug 1991–2020) über NOAA CoastWatch ERDDAP
Gebiet Niño 3.4: 5° S–5° N, 170° W–120° W, flächengewichtet (cos Breite)

Erstellt eine helle und eine dunkle Grafik (PNG und PDF) und eine Datentabelle:
  Grafiken/NINO34_jahr.png, Grafiken/NINO34_jahr_dunkel.png (+ .pdf)
  NINO34_tageswerte.txt  (Datum, Abweichung; laufendes Jahr und Vergleichsjahre)
Abgeschlossene Jahre werden einmal geladen und in nino34/ zwischengespeichert, das laufende
Jahr bei jedem Lauf neu (die letzten rund zwei Wochen sind bei OISST vorläufig).

Umgebungsvariablen (auf GitHub in der Steuerdatei):
  ERA5_FOLDER / NINO_FOLDER   Datenordner (auf GitHub der Datenordner der Webseite)

pip install requests pandas xarray netCDF4 matplotlib
"""
import os
import sys
import datetime as dt
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================= Einstellungen =================
JAHR = dt.date.today().year
VERGLEICH = [1997, 2015, 2023]   # Vergleichsjahre (starke El-Niño-Jahre)
HINTERGRUND = False              # True = alle Jahre ab 1982 dünn grau (einmaliger großer Download)
ORDNER = Path(os.environ.get("NINO_FOLDER") or os.environ.get(
    "ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ENSO"))
CACHE = ORDNER / "nino34"
GRAFIKEN = ORDNER / "Grafiken"
NAME = "NINO34_jahr"
TABELLE = ORDNER / "NINO34_tageswerte.txt"
# =================================================

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"

# ---------- Farben (wie die übrigen Grafiken: hell und dunkel) ----------
FARBEN = {
    False: dict(bg="white", box="white", rand="lightgray", text="#1d1d1f", achse="black",
                grau="gray", gitter="#e3e3e6", null="#555555", hinter="#d0d0d4", punkt_rand="white",
                jahr="#d7191c", vergleich={1997: "#f08a4b", 2015: "#b2182b", 2023: "#8c8c94"},
                warm="#b2182b", kalt="#2166ac"),
    True: dict(bg="#1e1e21", box="#2a2a2e", rand="#4a4a50", text="#ececee", achse="#cfcfcf",
               grau="#9a9aa2", gitter="#3a3a40", null="#cfcfcf", hinter="#4d4d55", punkt_rand="white",
               jahr="#ff3b4e", vergleich={1997: "#f4a582", 2015: "#d6604d", 2023: "#fddbc7"},
               warm="#e05563", kalt="#6aa6d8"),
}
MONATE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
DREI = ["Jan.", "Feb.", "Mär.", "Apr.", "Mai", "Jun.", "Jul.", "Aug.", "Sep.", "Okt.", "Nov.", "Dez."]


def tag_monat(d):
    return f"{d.day}. {DREI[d.month - 1]}"


def komma(x, plus=True):
    return (f"{x:+.2f}" if plus else f"{x:.2f}").replace(".", ",")


def lade_jahr(jahr: int) -> pd.Series:
    """Tageswerte eines Jahres laden (abgeschlossene Jahre aus dem Zwischenspeicher)."""
    import requests
    import xarray as xr

    CACHE.mkdir(parents=True, exist_ok=True)
    datei = CACHE / f"nino34_{jahr}.csv"
    laufend = jahr == dt.date.today().year
    if datei.exists() and not laufend:
        return pd.read_csv(datei, index_col=0, parse_dates=True).iloc[:, 0]

    ende = "last" if laufend else f"{jahr}-12-31T12:00:00Z"
    abfrage = f"anom[({jahr}-01-01T12:00:00Z):1:({ende})][(0.0)][(-5):1:(5)][(-170):1:(-120)]"
    print(f"Lade {jahr} …")
    r = requests.get(f"{ERDDAP}?{quote(abfrage)}", timeout=900)
    r.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(r.content)
    with xr.open_dataset(tmp.name) as ds:
        a = ds["anom"].squeeze(drop=True).load()
    Path(tmp.name).unlink(missing_ok=True)
    w = np.cos(np.deg2rad(a["latitude"]))
    s = a.weighted(w).mean(("latitude", "longitude")).to_series()

    s.index = pd.to_datetime(s.index).normalize()
    s.name = "anom"
    s = s.dropna()
    s.round(3).to_csv(datei)
    return s


def dummy(s: pd.Series) -> pd.Series:
    """Datum auf das Schaltjahr 2000 abbilden, damit alle Jahre übereinanderliegen."""
    s = s.copy()
    s.index = pd.to_datetime(s.index.strftime("2000-%m-%d"))
    return s


def zeichne(jahr, aktuell, vergleich, hintergrund, dunkel):
    F = FARBEN[dunkel]
    plt.rcParams.update({
        "figure.facecolor": F["bg"], "axes.facecolor": F["bg"], "savefig.facecolor": F["bg"],
        "axes.edgecolor": F["achse"], "axes.labelcolor": F["text"], "text.color": F["text"],
        "xtick.color": F["text"], "ytick.color": F["text"],
        "xtick.labelsize": 13, "ytick.labelsize": 13, "font.weight": "bold",
        "font.family": "DejaVu Sans",
    })
    boxstil = dict(boxstyle="round,pad=0.5", fc=F["box"], ec=F["rand"], alpha=0.95)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.12)

    # Titelblock wie bei den Temperaturgrafiken: groß, fett (Gebiet | Zeitraum | Datensatz), Bezug
    fig.text(0.5, 0.955, "Niño 3.4", ha="center", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.918, f"Äquatorialer Pazifik | {jahr} | NOAA OISST",
             ha="center", fontsize=19, fontweight="bold")
    fig.text(0.5, 0.885, "Abweichung der Meeresoberflächentemperatur vom Mittel 1991–2020",
             ha="center", fontsize=17, fontweight="normal")

    # El-Niño-/La-Niña-Schwellen
    for w, txt in [(0.5, "El Niño"), (1.0, "mäßig"), (1.5, "stark"), (2.0, "sehr stark")]:
        ax.axhline(w, color=F["warm"], ls=":", lw=1.3, alpha=0.4 + 0.15 * w)
        ax.text(pd.Timestamp("2000-12-29"), w + 0.03, txt, ha="right", va="bottom",
                fontsize=10, color=F["grau"])
    ax.axhline(-0.5, color=F["kalt"], ls=":", lw=1.3, alpha=0.7)
    ax.text(pd.Timestamp("2000-12-29"), -0.47, "La Niña", ha="right", va="bottom",
            fontsize=10, color=F["grau"])
    ax.axhline(0, color=F["null"], lw=0.9)

    for s in hintergrund.values():
        s = dummy(s)
        ax.plot(s.index, s.values, color=F["hinter"], lw=0.7, zorder=1)
    for j, s in vergleich.items():
        s = dummy(s)
        ax.plot(s.index, s.values, color=F["vergleich"].get(j, "#bbbbbb"), lw=1.6, label=str(j), zorder=2)
    sa = dummy(aktuell)
    ax.plot(sa.index, sa.values, color=F["jahr"], lw=3.0, label=str(jahr), zorder=4)

    # letzter Wert
    letzt_dat, letzt_wert = aktuell.index[-1], aktuell.iloc[-1]
    ax.scatter(sa.index[-1], letzt_wert, s=70, color=F["jahr"], ec=F["punkt_rand"], lw=1.2, zorder=5)

    # Infobox: aktueller Wert und Vergleich am selben Kalendertag
    tag = sa.index[-1]
    zeilen = [f"Niño 3.4 am {tag_monat(letzt_dat)} {letzt_dat.year}: {komma(letzt_wert)} K"]
    vgl = [f"{j}: {komma(dummy(s).get(tag, np.nan))} K" for j, s in vergleich.items()
           if not np.isnan(dummy(s).get(tag, np.nan))]
    if vgl:
        zeilen.append("Gleicher Tag  " + "  |  ".join(vgl))
    ax.text(0.012, 0.975, "\n".join(zeilen), transform=ax.transAxes, va="top",
            fontsize=13, bbox=boxstil, zorder=6)

    # Höchstwert der gezeigten Jahre
    alle = {**vergleich, jahr: aktuell}
    rj = max(alle, key=lambda j: alle[j].max())
    rd = alle[rj].idxmax()
    rw = alle[rj].max()
    ax.annotate(f"Höchster Tageswert {rj}\n{tag_monat(rd)} | {komma(rw)} K",
                xy=(dummy(alle[rj]).index[alle[rj].index.get_loc(rd)], rw),
                xytext=(0.80, 0.975), textcoords="axes fraction", ha="right", va="top",
                fontsize=13, bbox=boxstil, zorder=6,
                arrowprops=dict(arrowstyle="-", color=F["achse"], lw=1,
                                connectionstyle="angle,angleA=0,angleB=90"))

    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.80), fontsize=13,
                    facecolor=F["box"], edgecolor=F["rand"], framealpha=0.95, labelcolor=F["text"])
    leg.get_frame().set_boxstyle("round,pad=0.4")

    ax.set_xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-12-31"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: MONATE[mdates.num2date(x).month - 1]))
    werte = pd.concat([aktuell, *vergleich.values()])
    ax.set_ylim(min(-1.0, werte.min() - 0.3), werte.max() + 0.75)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",").replace("-", "−")))
    ax.set_ylabel("Temperaturabweichung [K]", fontsize=24, fontweight="bold")
    ax.set_xlabel("Monat", fontsize=24, fontweight="bold")
    ax.grid(axis="y", color=F["gitter"], lw=0.8)
    ax.set_axisbelow(True)

    fig.text(0.99, 0.012,
             "Datenquelle: NOAA OISST v2.1 (NCEI) über NOAA CoastWatch ERDDAP\n"
             "Gebiet 5° S–5° N, 170° W–120° W · Bezug 1991–2020 · eigene Berechnung\n"
             "Visualisierung: JohnnyLauge.bsky.social",
             ha="right", va="bottom", fontsize=11, color=F["grau"], fontweight="normal",
             linespacing=1.5)

    GRAFIKEN.mkdir(parents=True, exist_ok=True)
    basis = GRAFIKEN / (NAME + ("_dunkel" if dunkel else ""))
    fig.savefig(f"{basis}.png", dpi=128)
    fig.savefig(f"{basis}.pdf", dpi=128)
    plt.close(fig)
    print(f"Gespeichert: {basis}.png  (letzter Wert {letzt_dat:%d.%m.%Y}: {letzt_wert:+.2f} K)")


def schreibe_tabelle(aktuell, vergleich):
    """Tageswerte im einheitlichen TXT-Stil der Seite (die CSV macht Daten_Export.py)."""
    alle = pd.concat([*vergleich.values(), aktuell]).sort_index()
    alle = alle[~alle.index.duplicated(keep="last")]
    with open(TABELLE, "w", encoding="utf-8") as f:
        f.write("# Niño 3.4 – Abweichung der Meeresoberflächentemperatur (NOAA OISST v2.1), Tageswerte\n"
                f"# Gebiet 5° S–5° N, 170° W–120° W, flächengewichtet · Bezug 1991–2020 · "
                f"Jahre: {', '.join(str(j) for j in [*vergleich, JAHR])}\n"
                f"# Stand: {aktuell.index[-1]:%d.%m.%Y}\n"
                "#\n"
                f"{'Datum':>10}  {'Abw_K':>6}\n")
        for d, v in alle.items():
            f.write(f"{d:%d.%m.%Y}  {v:>+6.3f}\n")
    print(f"Tabelle gespeichert: {TABELLE}")


if __name__ == "__main__":
    try:
        aktuell = lade_jahr(JAHR)
        vergleich = {j: lade_jahr(j) for j in VERGLEICH}
        hintergrund = {j: lade_jahr(j) for j in range(1982, JAHR)
                       if HINTERGRUND and j not in VERGLEICH}
    except Exception as fehler:
        print(f"Niño 3.4: Daten nicht abrufbar ({type(fehler).__name__}: {str(fehler)[:300]})")
        sys.exit(0)                       # die übrige Webseite soll trotzdem entstehen
    for dunkel in (False, True):
        zeichne(JAHR, aktuell, vergleich, hintergrund, dunkel)
    schreibe_tabelle(aktuell, vergleich)
