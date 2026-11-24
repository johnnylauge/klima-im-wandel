"""
Europa (Landflächen 25° W–40° E, 34°–72° N): Tageswerte für die ERA5-Grafikskripte aufbereiten.

Quellen (gleiches Format: datum;tmittel;gitterpunkte):
  1. startdaten/Europa_Land_Tag.txt   - die Reihe vom PC (1940 bis zuletzt geladen)
  2. EUROPA_DATEN/Europa_Land_Tag.txt - die neuesten Tage, täglich von Europa_Download.py geladen
Wo sich beide überschneiden, gilt der Download (neuer).

Abweichung wie Copernicus:
  - Klimamittel 1991-2020 je Kalendertag als glatter Jahresgang (4 Harmonische), je Monat so nachgebessert,
    dass die Monatsmittel 1991-2020 genau erhalten bleiben (Monatswerte passen dann zu den Bulletins)
  - Abweichung von 1850-1900 = Abweichung von 1991-2020 + 1,43 K (Kopf der Copernicus-CSV)

Ergebnis: ERA5_europa_tag_1850.txt (date,2t,abw) - das lesen ERA5_Jahresmittel_Balken.py & Co.
mit ERA5_REGION=europa (Bezug 1850-1900, siehe region.py).

Zur Kontrolle im Protokoll: Jahresabweichungen gegen den Copernicus-Klimaindikator (ohne 29. Februar,
wie Copernicus) und die Monate des laufenden Jahres gegen die Bulletins.

Umgebungsvariablen:
  ERA5_FOLDER    hierhin kommt ERA5_europa_tag_1850.txt (auf GitHub _site/daten)
  EUROPA_DATEN   Ordner des täglichen Downloads (auf GitHub europa_daten)
"""

import os
import re
import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Copernicus\Europa")
DATEN = os.environ.get("EUROPA_DATEN", FOLDER)
STARTDATEN = os.path.join(HIER, "..", "startdaten")
ZIEL = os.path.join(FOLDER, "ERA5_europa_tag_1850.txt")
CSV_NAME = "time_series_of_annual_temperatures_1850-2025_wrt_1850-1900_europe.csv"
OFFSET_ERSATZ = 1.43           # 1991-2020 über 1850-1900, falls nicht im Kopf der CSV
HARMONISCHE = 4                # Klimamittel je Kalendertag: Jahresgang aus 4 Harmonischen (glatt, ohne Zacken)
# Copernicus-Bulletins (Europa Land, Abweichung 1991-2020) - nur zum Abgleich im Protokoll
BULLETIN = {2026: {1: -1.63, 2: -0.10, 3: 2.27, 4: 0.50, 5: 0.60, 6: 1.78, 7: 0.66, 8: 1.10,
                   9: 1.54, 10: 1.49}}


def lies(pfad):
    t = pd.read_csv(pfad, sep=";", parse_dates=["datum"], index_col="datum")["tmittel"]
    return t.dropna()


def finde(name, *ordner):
    for o in ordner:
        p = os.path.join(o, name)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------
# Tagesreihe zusammenführen
# ---------------------------------------------------------
teile = []
for o in dict.fromkeys([STARTDATEN, FOLDER, DATEN]):          # spätere Ordner haben Vorrang
    p = os.path.join(o, "Europa_Land_Tag.txt")
    if os.path.exists(p):
        s = lies(p)
        print(f"{p}: {s.index.min():%d.%m.%Y} bis {s.index.max():%d.%m.%Y}, {len(s)} Tage")
        teile.append(s)
if not teile:
    raise SystemExit("Keine Europa-Tagesreihe gefunden (startdaten/Europa_Land_Tag.txt)")
reihe = pd.concat(teile)
reihe = reihe[~reihe.index.duplicated(keep="last")].sort_index()     # erst Duplikate weg, dann sortieren
voll = pd.date_range(reihe.index.min(), reihe.index.max())
luecken = voll.difference(reihe.index)
if len(luecken):
    print(f"Hinweis: {len(luecken)} Tage fehlen, z. B. {', '.join(f'{d:%d.%m.%Y}' for d in luecken[:5])}")
basis = reihe["1991":"2020"]
if len(basis) < len(pd.date_range("1991-01-01", "2020-12-31")):
    raise SystemExit("1991-2020 nicht vollständig - Klimamittel nicht möglich")

# Abstand 1991-2020 -> 1850-1900 aus dem Kopf der Copernicus-CSV
csv = finde(CSV_NAME, STARTDATEN, FOLDER)
OFFSET, INDIKATOR = OFFSET_ERSATZ, None
if csv:
    text = open(csv, encoding="utf-8-sig").read()
    m = re.search(r"set at\s+([\d.]+)\s*C?\s+colder than 1991-2020", text)
    OFFSET = float(m.group(1)) if m else OFFSET_ERSATZ
    tab = pd.read_csv(csv, comment="#", index_col="Year")["ERA5"]
    INDIKATOR = tab[tab > -990]

# ---------------------------------------------------------
# Klimamittel 1991-2020 je Kalendertag
# ---------------------------------------------------------
kalender = pd.date_range("2000-01-01", "2000-12-31")                    # Schaltjahr als Kalender
roh = basis.groupby([basis.index.month, basis.index.day]).mean()
werte = np.array([roh[(d.month, d.day)] for d in kalender])
x = 2 * np.pi * np.arange(366) / 366
A = np.column_stack([np.ones(366)] + [f(k * x) for k in range(1, HARMONISCHE + 1) for f in (np.sin, np.cos)])
gew = np.where((kalender.month == 2) & (kalender.day == 29), 8 / 30, 1.0)   # 29.2. nur in 8 von 30 Jahren
koef = np.linalg.lstsq(A * np.sqrt(gew)[:, None], werte * np.sqrt(gew), rcond=None)[0]
glatt = pd.Series(A @ koef, index=pd.MultiIndex.from_arrays([kalender.month, kalender.day]))
# je Monat nachbessern: Mittel der Tage 1991-2020 minus Mittel des geglätteten Klimas über dieselben Tage
klima_b = glatt.reindex(pd.MultiIndex.from_arrays([basis.index.month, basis.index.day])).to_numpy()
korr = (pd.Series(basis.to_numpy() - klima_b, index=basis.index)
        .groupby(basis.index.month).mean())
klima = glatt + korr.reindex(glatt.index.get_level_values(0)).to_numpy()
tage = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
monat = basis.groupby(basis.index.month).mean()
print(f"Klimamittel 1991-2020: {np.average(monat, weights=tage):.3f} °C "
      f"(Jahresgang aus {HARMONISCHE} Harmonischen, Monatskorrektur höchstens {korr.abs().max():.3f} K)")

abw91 = reihe - klima.reindex(pd.MultiIndex.from_arrays([reihe.index.month, reihe.index.day])).to_numpy()
abw = abw91 + OFFSET

with open(ZIEL, "w", encoding="utf-8") as f:
    f.write("# Tagesmitteltemperatur Europa (Landflächen 25W-40E, 34-72N), ERA5; "
            f"abw = Abweichung von 1850-1900 (= 1991-2020 minus {OFFSET:.2f} K, wie Copernicus)\n")
    f.write("date,2t,abw\n")
    for d, t, a in zip(reihe.index, reihe.to_numpy(), abw.to_numpy()):
        f.write(f"{d:%Y-%m-%d},{t:.4f},{a:.4f}\n")
print(f"Gespeichert: {ZIEL} ({reihe.index.min():%d.%m.%Y} bis {reihe.index.max():%d.%m.%Y})")

# ---------------------------------------------------------
# Kontrolle
# ---------------------------------------------------------
ohne29 = abw91[~((abw91.index.month == 2) & (abw91.index.day == 29))]
jahre = ohne29.groupby(ohne29.index.year).agg(["mean", "size"])
jahre = jahre[jahre["size"] == 365]["mean"]
if INDIKATOR is not None:
    d = (jahre + OFFSET - INDIKATOR).dropna()
    if len(d):
        groesst = d.abs().idxmax()
        print(f"Abgleich mit dem Copernicus-Klimaindikator ({len(d)} Jahre): im Mittel {d.mean():+.3f} K, "
              f"größter Unterschied {d[groesst]:+.3f} K ({groesst})")
for jahr, ref in BULLETIN.items():
    a = abw91[str(jahr)] if str(jahr) in abw91.index.strftime("%Y") else pd.Series(dtype=float)
    if a.empty:
        continue
    print(f"Abgleich {jahr} mit den Copernicus-Bulletins (Abweichung von 1991-2020):")
    for m, w in a.groupby(a.index.month):
        ganz = len(w) == pd.Period(f"{jahr}-{m:02d}").days_in_month
        t = reihe[w.index].mean()
        zusatz = f"  Bulletin {ref[m]:+.2f}  Unterschied {w.mean() - ref[m]:+.2f}" if (m in ref and ganz) else ""
        print(f"  {m:02d}: {t:6.2f} °C  {w.mean():+.2f} K{'' if ganz else ' (bisher)'}{zusatz}")
