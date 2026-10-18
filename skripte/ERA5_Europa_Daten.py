"""
ERA5-Tagesmitteltemperatur für Europa (Landflächen) aus dem Copernicus Climate Data Store.

Ablauf:
  1. Lädt einmalig die Land-See-Maske von ERA5 für das Europa-Gebiet von C3S
     (34-72° N, 25° W-40° E) und merkt sie sich (ERA5_europa_lsm.nc).
  2. Lädt die Tagesmittel der 2-m-Temperatur ("ERA5 post-processed daily statistics") Monat für
     Monat und mittelt sie flächengewichtet über die Landflächen (Gitterpunkte mit mindestens 50 %
     Land, Gewicht cos(Breite)). Die großen Gitterdateien werden danach gleich wieder gelöscht,
     gespeichert wird nur ein Wert pro Tag: ERA5_europa_tag.txt (Datum, °C).
     Schon vorhandene Tage werden nicht noch einmal geladen - das Skript kann jederzeit
     abgebrochen und neu gestartet werden und macht dort weiter.
  3. Berechnet die Abweichungen zu den Bezugszeiträumen 1961-1990 und 1991-2020 (je Kalendertag
     ein Klimamittel, über 31 Tage geglättet) und schreibt sie für die Grafikskripte:
     ERA5_europa_tag_6190.txt und ERA5_europa_tag_9120.txt (date, 2t, abw).

Einmalig auf dem PC (1940 bis heute, dauert wegen der Warteschlange im CDS einige Stunden bis
etwa einen Tag, kann über Nacht laufen):
    python ERA5_Europa_Daten.py
Danach auf GitHub täglich: lädt nur die fehlenden neuesten Tage nach (ERA5 liegt etwa 5 Tage
hinter dem heutigen Datum).

Voraussetzungen:
  - Kostenloses Konto im Climate Data Store (https://cds.climate.copernicus.eu), dort die Lizenz
    der Datensätze akzeptiert ("ERA5 post-processed daily statistics on single levels" und
    "ERA5 hourly data on single levels").
  - API-Schlüssel: lokal in der Datei %USERPROFILE%\\.cdsapirc
        url: https://cds.climate.copernicus.eu/api
        key: <dein-schlüssel>
    auf GitHub als Geheimnis CDSAPI_KEY (die Steuerdatei setzt CDSAPI_URL und CDSAPI_KEY).
  - Pakete: pip install cdsapi xarray netCDF4

Umgebungsvariablen:
  ERA5_FOLDER          Datenordner (auf GitHub der Datenordner der Webseite)
  EUROPA_MAX_MONATE    höchstens so viele Monate pro Lauf laden (Standard: alle); die neuesten
                       zwei Monate kommen immer dazu, sonst wird von 1940 an aufgefüllt
  EUROPA_MAX_MINUTEN   nach so vielen Minuten keine neuen Monate mehr anfangen (für GitHub)
  EUROPA_PARALLEL      gleichzeitige Anfragen an den CDS (Standard 3)
  EUROPA_OHNE_DOWNLOAD=1  nichts laden, nur die Abweichungen neu berechnen
"""

import os
import time
import zipfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# =========================================================
# EINSTELLUNGEN
# =========================================================
START_JAHR = 1940
GEBIET = [72, -25, 34, 40]          # Nord, West, Süd, Ost - Europa-Gebiet von C3S
LAND_AB = 0.5                       # Gitterpunkt zählt als Land ab 50 % Landanteil
FREQUENZ = "1_hourly"               # Tagesmittel aus stündlichen Werten (wie Climate Pulse)
LATENZ_TAGE = 5                     # ERA5 liegt etwa so viele Tage zurück
BEZUEGE = {"6190": (1961, 1990), "9120": (1991, 2020)}
GLAETTUNG = 31                      # Klimamittel je Kalendertag über so viele Tage geglättet

FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")   # wie die Grafikskripte
TAGE_DATEI = os.path.join(FOLDER, "ERA5_europa_tag.txt")
MASKE_DATEI = os.path.join(FOLDER, "ERA5_europa_lsm.nc")
MAX_MONATE = int(os.environ.get("EUROPA_MAX_MONATE", "0")) or None
PARALLEL = int(os.environ.get("EUROPA_PARALLEL", "3"))
MAX_MINUTEN = float(os.environ.get("EUROPA_MAX_MINUTEN", "0")) or None
OHNE_DOWNLOAD = os.environ.get("EUROPA_OHNE_DOWNLOAD") == "1"

DATENSATZ = "derived-era5-single-levels-daily-statistics"
os.makedirs(FOLDER, exist_ok=True)
SCHLOSS = threading.Lock()


# =========================================================
# HILFSFUNKTIONEN
# =========================================================
def client():
    import cdsapi
    # wenige, kurze Wiederholungen: ohne diese Einstellung wartet cdsapi bei Serverfehlern bis zu
    # 500-mal je 2 Minuten - der Lauf würde stundenlang hängen
    return cdsapi.Client(quiet=True, progress=False, retry_max=3, sleep_max=60, timeout=600)


def oeffne_netcdf(pfad):
    """Öffnet eine NetCDF-Datei - auch wenn der CDS sie als ZIP liefert."""
    import xarray as xr
    with open(pfad, "rb") as f:
        kopf = f.read(4)
    if kopf[:2] == b"PK":                              # ZIP: erste .nc darin auspacken
        with zipfile.ZipFile(pfad) as z:
            name = next(n for n in z.namelist() if n.endswith(".nc"))
            ziel = pfad + "_" + os.path.basename(name)
            with open(ziel, "wb") as f:
                f.write(z.read(name))
        pfad = ziel
    with xr.open_dataset(pfad) as ds:                  # Datei gleich wieder schließen (Windows)
        return ds.load()


def koordinaten(ds):
    zeit = next(n for n in ("valid_time", "time", "date") if n in ds.coords or n in ds.dims)
    breite = next(n for n in ("latitude", "lat") if n in ds.coords)
    laenge = next(n for n in ("longitude", "lon") if n in ds.coords)
    return zeit, breite, laenge


def lade_maske():
    """Land-See-Maske (einmal laden, dann aus der Datei)."""
    if not os.path.exists(MASKE_DATEI):
        print("Lade die Land-See-Maske ...")
        client().retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": ["land_sea_mask"],
            "year": ["2020"], "month": ["01"], "day": ["01"], "time": ["00:00"],
            "area": GEBIET, "data_format": "netcdf", "download_format": "unarchived",
        }, MASKE_DATEI)
    ds = oeffne_netcdf(MASKE_DATEI)
    var = list(ds.data_vars)[0]
    zeit, breite, laenge = koordinaten(ds)
    lsm = ds[var].isel({zeit: 0}) if zeit in ds[var].dims else ds[var]
    gewicht = np.cos(np.deg2rad(lsm[breite])) * (lsm >= LAND_AB)
    print(f"Landpunkte im Europa-Gebiet: {int((lsm >= LAND_AB).sum())}")
    return gewicht.drop_vars([c for c in gewicht.coords if c not in (breite, laenge)])


def lies_tage():
    if not os.path.exists(TAGE_DATEI):
        return pd.Series(dtype=float)
    t = pd.read_csv(TAGE_DATEI, comment="#", parse_dates=["date"])
    return t.set_index("date")["2t"]


def schreibe_tage(reihe):
    reihe = reihe[~reihe.index.duplicated(keep="last")].sort_index()
    tmp = TAGE_DATEI + ".neu"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Tagesmitteltemperatur Europa (Landflächen 34-72 N, 25 W-40 E), ERA5 daily statistics, °C\n")
        f.write(f"# Stand: {reihe.index.max():%d.%m.%Y}\n")
        f.write("date,2t\n")
        for d, v in reihe.items():
            f.write(f"{d:%Y-%m-%d},{v:.4f}\n")
    os.replace(tmp, TAGE_DATEI)


def lade_monat(jahr, monat, tage, gewicht, versuche=3):
    """Lädt die Tagesmittel eines Monats und gibt das Landflächenmittel je Tag zurück (°C)."""
    with tempfile.TemporaryDirectory() as tmp:
        ziel = os.path.join(tmp, f"t2m_{jahr}_{monat:02d}.nc")
        anfrage = {
            "product_type": "reanalysis", "variable": ["2m_temperature"],
            "year": str(jahr), "month": [f"{monat:02d}"], "day": [f"{t:02d}" for t in tage],
            "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": FREQUENZ,
            "area": GEBIET,
        }
        for versuch in range(1, versuche + 1):
            try:
                client().retrieve(DATENSATZ, anfrage, ziel)
                break
            except Exception as fehler:                       # Warteschlange, Netz, Wartung ...
                if versuch == versuche:
                    raise
                print(f"  {jahr}-{monat:02d}: Fehler ({str(fehler)[:120]}) - neuer Versuch in 2 Minuten")
                time.sleep(120)
        ds = oeffne_netcdf(ziel)
        var = list(ds.data_vars)[0]
        zeit, breite, laenge = koordinaten(ds)
        feld = ds[var]
        g = gewicht.reindex_like(feld.isel({zeit: 0}), method="nearest", tolerance=1e-3).fillna(0)
        mittel = (feld * g).sum((breite, laenge)) / (g * feld.notnull()).sum((breite, laenge))
        werte = mittel.to_series() - 273.15
        werte.index = pd.to_datetime(werte.index).normalize()
        return werte.dropna()


def lade_flexibel(jahr, monat, tage, gewicht, neuester):
    """Beim neuesten Monat fehlen die letzten Tage im CDS manchmal noch -
    dann bis zu drei Tage weniger anfragen."""
    for weniger in range(4 if neuester else 1):
        try:
            letzter_versuch = weniger == (3 if neuester else 0)
            return lade_monat(jahr, monat, tage[:len(tage) - weniger], gewicht,
                              versuche=3 if letzter_versuch else 1)
        except Exception:
            if weniger == (3 if neuester else 0) or len(tage) - weniger <= 1:
                raise
            print(f"  {jahr}-{monat:02d}: neueste Tage noch nicht verfügbar - ein Tag weniger")


# =========================================================
# 1.+2. DOWNLOAD (nur fehlende Tage)
# =========================================================
def herunterladen():
    vorhanden = lies_tage()
    letzter_tag = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=LATENZ_TAGE)
    auftraege = []
    for monat_start in pd.date_range(f"{START_JAHR}-01-01", letzter_tag, freq="MS"):
        ende = min(monat_start + pd.offsets.MonthEnd(0), letzter_tag)
        fehlend = [d for d in pd.date_range(monat_start, ende) if d not in vorhanden.index]
        if fehlend:
            auftraege.append((monat_start.year, monat_start.month, [d.day for d in fehlend]))
    if MAX_MONATE and len(auftraege) > MAX_MONATE:
        # die neuesten zwei Monate immer, dazu die ältesten fehlenden (so füllt sich die Reihe von 1940 an)
        auftraege = auftraege[:max(MAX_MONATE - 2, 0)] + auftraege[-2:]
    if not auftraege:
        print("Europa: alle Tage bis", f"{letzter_tag:%d.%m.%Y}", "sind schon vorhanden.")
        return
    print(f"Europa: {len(auftraege)} Monat(e) zu laden, {PARALLEL} Anfragen gleichzeitig ...")
    gewicht = lade_maske()
    reihe = vorhanden.copy()
    fertig = 0
    beginn = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        neuester = auftraege[-1][:2]
        zukunft = {pool.submit(lade_flexibel, j, m, t, gewicht, (j, m) == neuester): (j, m)
                   for j, m, t in auftraege}
        for z in as_completed(zukunft):
            if MAX_MINUTEN and time.time() - beginn > MAX_MINUTEN * 60:
                offen = sum(f.cancel() for f in zukunft)          # noch nicht begonnene Monate auslassen
                print(f"Zeitlimit erreicht - {offen} Monat(e) beim nächsten Lauf")
                break
            j, m = zukunft[z]
            try:
                werte = z.result()
            except Exception as fehler:
                # neueste Tage oft noch nicht verfügbar -> beim nächsten Lauf erneut
                print(f"  {j}-{m:02d}: nicht geladen ({type(fehler).__name__}: {str(fehler)[:600]})")
                continue
            with SCHLOSS:
                reihe = pd.concat([reihe, werte])
                schreibe_tage(reihe)                       # nach jedem Monat sichern
                fertig += 1
            print(f"  {j}-{m:02d}: {len(werte)} Tage  ({fertig}/{len(auftraege)})")


# =========================================================
# 3. ABWEICHUNGEN ZU DEN BEZUGSZEITRÄUMEN
# =========================================================
def klimamittel(reihe, von, bis):
    """Mittel je Kalendertag (366 Tage, 29.2. eingeschlossen), über GLAETTUNG Tage geglättet."""
    basis = reihe[(reihe.index.year >= von) & (reihe.index.year <= bis)]
    tage = pd.date_range("2000-01-01", "2000-12-31")                  # Schaltjahr als Kalender
    roh = basis.groupby([basis.index.month, basis.index.day]).mean()
    werte = np.array([roh.get((d.month, d.day), np.nan) for d in tage])
    if np.isnan(werte[59]):                                           # 29.2. ohne Werte
        werte[59] = (werte[58] + werte[60]) / 2
    h = GLAETTUNG // 2
    rund = np.concatenate([werte[-h:], werte, werte[:h]])             # Jahreswechsel überbrücken
    glatt = np.convolve(rund, np.ones(GLAETTUNG) / GLAETTUNG, mode="valid")
    return {(d.month, d.day): v for d, v in zip(tage, glatt)}


def abweichungen():
    reihe = lies_tage()
    if reihe.empty:
        print("Keine Tageswerte vorhanden - erst herunterladen.")
        return
    for kurz, (von, bis) in BEZUEGE.items():
        if reihe.index.min().year > von or reihe.index.max().year < bis:
            print(f"Bezug {von}-{bis}: Daten noch unvollständig - übersprungen")
            continue
        klima = klimamittel(reihe, von, bis)
        datei = os.path.join(FOLDER, f"ERA5_europa_tag_{kurz}.txt")
        with open(datei, "w", encoding="utf-8") as f:
            f.write(f"# Tagesmitteltemperatur Europa (Landflächen), ERA5; abw = Abweichung zum Mittel {von}-{bis}\n")
            f.write("date,2t,abw\n")
            for d, v in reihe.items():
                f.write(f"{d:%Y-%m-%d},{v:.4f},{v - klima[(d.month, d.day)]:.4f}\n")
        print(f"Abweichungen zu {von}-{bis} gespeichert unter:\n{datei}")


if not OHNE_DOWNLOAD:
    try:
        herunterladen()
    except Exception as fehler:      # z. B. CDS nicht erreichbar oder kein Schlüssel: mit den vorhandenen Tagen weiter
        print(f"Download nicht möglich: {type(fehler).__name__}: {str(fehler)[:1500]}")
        print("Rechne mit den vorhandenen Tageswerten weiter.")
abweichungen()
