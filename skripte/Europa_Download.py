"""
Europa (Landflächen 25° W–40° E, 34°–72° N): ERA5-Tagesmittel aus dem Copernicus Climate Data Store,
gerechnet wie Copernicus (Climate Bulletin) - für das laufende Jahr auf GitHub.

Läuft täglich um 06 UTC (Steuerdatei europa_daten.yml). Je Lauf:
  1. einmalig die Land-See-Maske des Gebiets                    -> ERA5_europa_lsm.nc
  2. einmalig das Klimamittel 1991-2020 aus den ERA5-Monatsmitteln -> Europa_Land_Klima_9120.txt
  3. die fehlenden Tage ab EUROPA_START (Standard 1. Januar 2026) bis heute minus 6 Tage,
     je Monat eine Anfrage                                         -> Europa_Land_Tag.txt
Gebietsmittel: flächengewichtet (Kosinus der Breite), nur Gitterpunkte mit Landanteil >= 0,5.
Tagesmittel: derived-era5-single-levels-daily-statistics (daily_mean, UTC, aus Stundenwerten).

Die Anfragen laufen nacheinander (ECMWF: keine parallelen oder massenhaften Anfragen). Die Nummer
der offenen Anfrage steht in offene_anfrage.json: wartet sie beim Ende des Laufs noch beim CDS,
holt der nächste Lauf sie ab und behält ihren Platz in der Warteschlange.

Die Tagesreihe hat dasselbe Format wie die auf dem PC (era5_regionen_auswertung.py):
    datum;tmittel;gitterpunkte
Später kann die vollständige Reihe vom PC einfach daneben gelegt werden (EUROPA_PC_REIHE).

Umgebungsvariablen:
  ERA5_FOLDER          Datenordner (auf GitHub: europa_daten, im Zwischenspeicher "europa-")
  CDSAPI_URL, CDSAPI_KEY   Zugang zum CDS (auf GitHub als Geheimnis CDSAPI_KEY)
  EUROPA_START         erster Tag der Tagesreihe (Standard 2026-01-01)
  EUROPA_MAX_MINUTEN   so lange höchstens laufen und warten (Standard 50)
"""

import os
import sys
import json
import time
import shutil
import zipfile
import tempfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

# =========================================================
# EINSTELLUNGEN
# =========================================================
GEBIET = [72, -25, 34, 40]            # Nord, West, Süd, Ost (Copernicus: 25° W–40° E, 34°–72° N)
LAND_AB = 0.5                         # Gitterpunkt zählt als Land ab 50 % Landanteil
VERZUG_TAGE = 6                       # ERA5 im CDS liegt rund 5 Tage zurück
START = date.fromisoformat(os.environ.get("EUROPA_START", "2026-01-01"))
MAX_MINUTEN = float(os.environ.get("EUROPA_MAX_MINUTEN", "50"))
PRUEFEN_SEKUNDEN = 60

FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Copernicus\Europa")
MASKE = os.path.join(FOLDER, "ERA5_europa_lsm.nc")
KLIMA = os.path.join(FOLDER, "Europa_Land_Klima_9120.txt")
TAGE = os.path.join(FOLDER, "Europa_Land_Tag.txt")
OFFEN = os.path.join(FOLDER, "offene_anfrage.json")

# Monatsbulletins von Copernicus (Europa Land, Abweichung 1991-2020) - nur zum Abgleich im Protokoll
BULLETIN_2026 = {1: -1.63, 2: -0.10, 3: 2.27, 4: 0.50, 5: 0.60, 6: 1.78, 7: 0.66, 8: 1.10, 9: 1.54, 10: 1.49}

os.makedirs(FOLDER, exist_ok=True)


# =========================================================
# CDS
# =========================================================
def client():
    url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
    key = os.environ.get("CDSAPI_KEY")
    if not key:
        rc = os.path.expanduser("~/.cdsapirc")
        if os.path.exists(rc):
            for z in open(rc, encoding="utf-8"):
                if z.strip().startswith("key:"):
                    key = z.split(":", 1)[1].strip()
                elif z.strip().startswith("url:"):
                    url = z.split(":", 1)[1].strip()
    if not key:
        sys.exit("Kein CDS-Schlüssel: auf GitHub das Geheimnis CDSAPI_KEY anlegen.")
    for modul, klasse in (("ecmwf.datastores", "Client"), ("datapi", "ApiClient"), ("cads_api_client", "ApiClient")):
        try:
            mod = __import__(modul, fromlist=[klasse])
            return getattr(mod, klasse)(url=url, key=key)
        except Exception:
            continue
    sys.exit("Kein CDS-Client gefunden: pip install ecmwf-datastores-client")


def fehlertext(f):
    t = str(f)
    a = getattr(f, "response", None)
    if a is not None:
        try:
            t += " | " + a.text[:300]
        except Exception:
            pass
    return t


def anfrage_fuer(a):
    if a["art"] == "maske":
        return "reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": ["land_sea_mask"],
            "year": ["2020"], "month": ["01"], "day": ["01"], "time": ["00:00"],
            "area": GEBIET, "data_format": "netcdf", "download_format": "unarchived"}
    if a["art"] == "klima":
        return "reanalysis-era5-single-levels-monthly-means", {
            "product_type": ["monthly_averaged_reanalysis"], "variable": ["2m_temperature"],
            "year": [str(j) for j in range(1991, 2021)], "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"], "area": GEBIET, "data_format": "netcdf", "download_format": "unarchived"}
    return "derived-era5-single-levels-daily-statistics", {
        "product_type": "reanalysis", "variable": ["2m_temperature"],
        "year": str(a["jahr"]), "month": [f"{a['monat']:02d}"], "day": [f"{t:02d}" for t in a["tage"]],
        "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": "1_hourly",
        "area": GEBIET}


def name_von(a):
    return {"maske": "Landmaske", "klima": "Klimamittel 1991-2020"}.get(a["art"]) or f"{a['jahr']}-{a['monat']:02d}"


# =========================================================
# NETCDF UND GEBIETSMITTEL
# =========================================================
def oeffne(pfad):
    import xarray as xr
    with open(pfad, "rb") as f:
        zip_ = f.read(2) == b"PK"
    if zip_:
        with zipfile.ZipFile(pfad) as z:
            name = next(n for n in z.namelist() if n.endswith(".nc"))
            ziel = pfad + "_inhalt.nc"
            with open(ziel, "wb") as f:
                f.write(z.read(name))
        pfad = ziel
    with xr.open_dataset(pfad) as ds:
        return ds.load()


def feld_aus(ds):
    if "valid_time" in ds.variables:
        ds = ds.rename({"valid_time": "time"})
    for dim in ("number", "expver"):
        if dim in ds.dims:
            ds = ds.isel({dim: 0})
    name = next(n for n in ("t2m", "2t", "lsm") if n in ds.data_vars)
    f = ds[name]
    f = f.assign_coords(latitude=f["latitude"].round(4), longitude=f["longitude"].round(4))
    return f.sortby("latitude").sortby("longitude")


def gewichte():
    m = feld_aus(oeffne(MASKE))
    if "time" in m.dims:
        m = m.isel(time=0)
    land = (m >= LAND_AB)
    g = np.cos(np.deg2rad(m["latitude"])) * land
    return g, int(land.sum())


def gebietsmittel(pfad, g):
    """Landflächenmittel je Zeitschritt in °C."""
    f = feld_aus(oeffne(pfad))
    f = f.sel(latitude=g["latitude"], longitude=g["longitude"], method="nearest")
    f = f.assign_coords(latitude=g["latitude"], longitude=g["longitude"])
    w = g * f.notnull()
    m = (f.fillna(0) * w).sum(("latitude", "longitude")) / w.sum(("latitude", "longitude"))
    s = m.to_series()
    if s.mean() > 100:                                   # Kelvin
        s = s - 273.15
    s.index = pd.to_datetime(s.index).normalize()
    return s.dropna()


# =========================================================
# DATEIEN
# =========================================================
def lies_tage():
    if not os.path.exists(TAGE):
        return pd.DataFrame(columns=["tmittel", "gitterpunkte"], index=pd.DatetimeIndex([], name="datum"))
    return pd.read_csv(TAGE, sep=";", parse_dates=["datum"], index_col="datum")


def schreibe_tage(tab):
    tab = tab[~tab.index.duplicated(keep="last")].sort_index()
    tmp = TAGE + ".neu"
    tab.round({"tmittel": 3}).to_csv(tmp, sep=";", date_format="%Y-%m-%d", index_label="datum")
    os.replace(tmp, TAGE)


def lies_offen():
    try:
        with open(OFFEN, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def schreibe_offen(a):
    if a is None:
        if os.path.exists(OFFEN):
            os.remove(OFFEN)
        return
    with open(OFFEN, "w", encoding="utf-8") as f:
        json.dump(a, f)


# =========================================================
# ABLAUF
# =========================================================
def naechste_aufgabe(letzter_tag):
    if not os.path.exists(MASKE):
        return {"art": "maske"}
    if not os.path.exists(KLIMA):
        return {"art": "klima"}
    vorhanden = set(lies_tage().index.date)
    for start in pd.date_range(START.replace(day=1), letzter_tag, freq="MS"):
        ende = min((start + pd.offsets.MonthEnd(0)).date(), letzter_tag)
        tage = [d.date() for d in pd.date_range(max(start.date(), START), ende)]
        fehlend = [d for d in tage if d not in vorhanden]
        if fehlend:
            # zusammenhängend vom ersten fehlenden Tag bis Monatsende (bzw. letzter verfügbarer Tag)
            erster = min(fehlend)
            return {"art": "tage", "jahr": start.year, "monat": start.month,
                    "tage": [d.day for d in tage if d >= erster]}
    return None


def verarbeiten(a, pfad):
    if a["art"] == "maske":
        shutil.move(pfad, MASKE)            # Zwischenordner kann auf einem anderen Laufwerk liegen
        g, n = gewichte()
        print(f"  Landmaske gespeichert: {n} Landpunkte im Gebiet")
        return
    g, n = gewichte()
    s = gebietsmittel(pfad, g)
    if a["art"] == "klima":
        mon = s.groupby(s.index.month).mean()
        with open(KLIMA, "w", encoding="utf-8") as f:
            f.write("# Europa (Land, 25W-40E, 34-72N): Klimamittel 1991-2020 aus den ERA5-Monatsmitteln, °C\n")
            f.write("monat;tmittel\n")
            for m, v in mon.items():
                f.write(f"{m};{v:.4f}\n")
        tage = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        print(f"  Klimamittel 1991-2020 gespeichert, Jahresmittel {np.average(mon, weights=tage):.2f} °C")
        return
    tab = lies_tage()
    neu = pd.DataFrame({"tmittel": s, "gitterpunkte": n})
    neu.index.name = "datum"
    schreibe_tage(pd.concat([tab, neu]))
    print(f"  {name_von(a)}: {len(s)} Tage gespeichert, Mittel {s.mean():.2f} °C")


def abgleich():
    """Monatsabweichungen des laufenden Jahres gegen die Copernicus-Bulletins (nur Protokoll)."""
    if not (os.path.exists(KLIMA) and os.path.exists(TAGE)):
        return
    klima = pd.read_csv(KLIMA, sep=";", comment="#", index_col="monat")["tmittel"]
    t = lies_tage()["tmittel"]
    t = t[t.index.year == 2026]
    if t.empty:
        return
    print("Abgleich 2026 mit den Copernicus-Bulletins (Abweichung von 1991-2020):")
    for m, werte in t.groupby(t.index.month):
        voll = len(werte) == pd.Period(f"2026-{m:02d}").days_in_month
        abw = werte.mean() - klima[m]
        ref = BULLETIN_2026.get(m)
        zusatz = f"  Bulletin {ref:+.2f}  Unterschied {abw - ref:+.2f}" if (ref is not None and voll) else ""
        print(f"  {m:02d}: {abw:+.2f} K{'' if voll else ' (bisher)'}{zusatz}")


def main():
    letzter_tag = date.today() - timedelta(days=VERZUG_TAGE)
    cl = client()
    beginn = time.time()
    fehler = 0
    a = lies_offen()
    if a:
        print(f"Offene Anfrage vom letzten Lauf: {name_von(a)} ({a['id']})")
    while time.time() - beginn < MAX_MINUTEN * 60:
        if a is None:
            a = naechste_aufgabe(letzter_tag)
            if a is None:
                print(f"Alle Tage bis {letzter_tag:%d.%m.%Y} vorhanden.")
                break
            datensatz, req = anfrage_fuer(a)
            try:
                a["id"] = cl.submit(datensatz, req).request_id
            except Exception as f:
                print(f"  {name_von(a)}: Anfrage nicht angenommen ({fehlertext(f)[:300]})")
                break
            schreibe_offen(a)
            print(f"  {name_von(a)}: Anfrage gestellt ({a['id']})")
        try:
            remote = cl.get_remote(a["id"])
            status = remote.status
        except Exception as f:
            if "404" in fehlertext(f) or "not found" in fehlertext(f).lower():
                print(f"  {name_von(a)}: Anfrage gibt es beim CDS nicht mehr - wird neu gestellt")
                schreibe_offen(None)
                a = None
                continue
            print(f"  Statusabfrage gescheitert ({fehlertext(f)[:200]})")
            time.sleep(PRUEFEN_SEKUNDEN)
            continue
        if status == "successful":
            with tempfile.TemporaryDirectory() as tmp:
                pfad = os.path.join(tmp, "lieferung.nc")
                remote.download(pfad)
                verarbeiten(a, pfad)
            schreibe_offen(None)
            a, fehler = None, 0
            continue
        if status in ("failed", "rejected", "dismissed", "deleted"):
            print(f"  {name_von(a)}: vom CDS abgelehnt ({status})")
            schreibe_offen(None)
            fehler += 1
            if a["art"] == "tage" and len(a["tage"]) > 3:
                # am aktuellen Rand fehlen beim CDS evtl. noch Tage: kürzer versuchen
                a = {k: v for k, v in a.items() if k != "id"}
                a["tage"] = a["tage"][:-3]
                datensatz, req = anfrage_fuer(a)
                if fehler >= 3:
                    print("Drei Ablehnungen in Folge - Abbruch, nächster Versuch beim nächsten Lauf.")
                    break
                try:
                    a["id"] = cl.submit(datensatz, req).request_id
                    schreibe_offen(a)
                    print(f"  {name_von(a)}: neu gestellt mit {len(a['tage'])} Tagen ({a['id']})")
                    continue
                except Exception as f:
                    print(f"  Anfrage nicht angenommen ({fehlertext(f)[:300]})")
                    break
            a = None
            if fehler >= 3:
                print("Drei Ablehnungen in Folge - Abbruch.")
                break
            continue
        time.sleep(PRUEFEN_SEKUNDEN)                     # accepted / running
    else:
        if a:
            print(f"Zeitlimit: {name_von(a)} wartet weiter beim CDS und wird beim nächsten Lauf abgeholt.")
    t = lies_tage()
    if len(t):
        print(f"Tagesreihe: {t.index.min():%d.%m.%Y} bis {t.index.max():%d.%m.%Y}, {len(t)} Tage")
    abgleich()


if __name__ == "__main__":
    main()
