"""
ERA5-Meeresoberflächentemperatur im Gebiet Niño 3.4 (5° S–5° N, 170° W–120° W) aus dem
Copernicus Climate Data Store - für die ERA5-Fassung der Niño-3.4-Grafik.

Lädt
  1. einmalig das Klimamittel 1991–2020 aus den ERA5-Monatsmitteln (eine einzige kleine Anfrage):
     ERA5_nino34_klima.txt  (Monat, SST in °C)
  2. die Tagesmittel der SST ("ERA5 post-processed daily statistics") für das laufende Jahr und die
     Vergleichsjahre 1997, 2015, 2023 - je Monat eine Anfrage:
     ERA5_nino34_tag.txt  (Datum, SST in °C, flächengewichtet mit cos Breite)

Wie beim Europa-Download werden Anfragen nur abgeschickt und gemerkt (ERA5_nino34_offen.json);
jeder Lauf holt fertige ab und schickt neue nach. So geht bei voller CDS-Warteschlange nichts verloren.

Umgebungsvariablen:
  ERA5_FOLDER        Datenordner
  NINO_MAX_MINUTEN   so lange höchstens warten (Standard 20)
  NINO_MAX_OFFEN     so viele Anfragen gleichzeitig beim CDS (Standard 2)
Schlüssel wie beim Europa-Download: CDSAPI_URL / CDSAPI_KEY oder ~/.cdsapirc
"""
import os
import json
import time
import zipfile
import tempfile

import numpy as np
import pandas as pd

GEBIET = [5, -170, -5, -120]                    # Nord, West, Süd, Ost
VERGLEICH = [1997, 2015, 2023]
LATENZ_TAGE = 6
FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ENSO")
TAGE_DATEI = os.path.join(FOLDER, "ERA5_nino34_tag.txt")
KLIMA_DATEI = os.path.join(FOLDER, "ERA5_nino34_klima.txt")
OFFEN_DATEI = os.path.join(FOLDER, "ERA5_nino34_offen.json")
MAX_MINUTEN = float(os.environ.get("NINO_MAX_MINUTEN", "20"))
MAX_OFFEN = int(os.environ.get("NINO_MAX_OFFEN", "2"))
TAGES_DATENSATZ = "derived-era5-single-levels-daily-statistics"
MONATS_DATENSATZ = "reanalysis-era5-single-levels-monthly-means"
os.makedirs(FOLDER, exist_ok=True)


# ---------------- Hilfsfunktionen ----------------
def fehlertext(fehler):
    text = str(fehler)
    antwort = getattr(fehler, "response", None)
    if antwort is not None:
        try:
            text += " | " + antwort.text[:400]
        except Exception:
            pass
    return text


def client():
    """Client, der Anfragen abschicken und später abholen kann (gehört zu cdsapi dazu)."""
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
    for modul, klasse in (("ecmwf.datastores", "Client"), ("datapi", "ApiClient"), ("cads_api_client", "ApiClient")):
        try:
            mod = __import__(modul, fromlist=[klasse])
            return getattr(mod, klasse)(url=url, key=key)
        except Exception:
            continue
    return None


def oeffne_netcdf(pfad):
    import xarray as xr
    with open(pfad, "rb") as f:
        kopf = f.read(4)
    if kopf[:2] == b"PK":
        with zipfile.ZipFile(pfad) as z:
            name = next(n for n in z.namelist() if n.endswith(".nc"))
            ziel = pfad + "_" + os.path.basename(name)
            with open(ziel, "wb") as f:
                f.write(z.read(name))
        pfad = ziel
    with xr.open_dataset(pfad) as ds:
        return ds.load()


def gebietsmittel(pfad):
    """Flächengewichtetes Mittel (cos Breite) je Zeitschritt in °C."""
    ds = oeffne_netcdf(pfad)
    var = list(ds.data_vars)[0]
    feld = ds[var]
    zeit = next(n for n in ("valid_time", "time", "date") if n in feld.dims)
    breite = next(n for n in ("latitude", "lat") if n in feld.dims)
    laenge = next(n for n in ("longitude", "lon") if n in feld.dims)
    g = np.cos(np.deg2rad(feld[breite])) * feld.notnull()
    mittel = (feld.fillna(0) * g).sum((breite, laenge)) / g.sum((breite, laenge))
    s = mittel.to_series() - 273.15
    s.index = pd.to_datetime(s.index)
    return s.dropna().rename(None)


def lies_tage():
    if not os.path.exists(TAGE_DATEI):
        return pd.Series(dtype=float)
    t = pd.read_csv(TAGE_DATEI, comment="#", parse_dates=["date"])
    return t.set_index("date")["sst"]


def schreibe_tage(reihe):
    reihe = reihe[~reihe.index.duplicated(keep="last")].sort_index()
    with open(TAGE_DATEI, "w", encoding="utf-8") as f:
        f.write("# Meeresoberflächentemperatur Niño 3.4 (5S-5N, 170W-120W), ERA5 daily statistics, °C\n")
        f.write("date,sst\n")
        for d, v in reihe.items():
            f.write(f"{d:%Y-%m-%d},{v:.4f}\n")


def schreibe_klima(s):
    werte = s.groupby(s.index.month).mean()
    with open(KLIMA_DATEI, "w", encoding="utf-8") as f:
        f.write("# Klimamittel 1991-2020 der SST Niño 3.4 aus ERA5-Monatsmitteln, °C\n")
        f.write("monat,sst\n")
        for m, v in werte.items():
            f.write(f"{m},{v:.4f}\n")
    print(f"Klimamittel 1991-2020 gespeichert: {KLIMA_DATEI}")


# ---------------- Aufträge ----------------
def auftraege():
    vorhanden = lies_tage()
    letzter_tag = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=LATENZ_TAGE)
    liste = []
    if not os.path.exists(KLIMA_DATEI):
        liste.append(dict(art="klima", name="Klimamittel 1991-2020"))
    for jahr in [letzter_tag.year, *VERGLEICH]:
        for m in range(1, 13):
            anfang = pd.Timestamp(jahr, m, 1)
            if anfang > letzter_tag:
                break
            ende = min(anfang + pd.offsets.MonthEnd(0), letzter_tag)
            tage = pd.date_range(anfang, ende)
            if all(d in vorhanden.index for d in tage):
                continue
            liste.append(dict(art="tage", name=f"{jahr}-{m:02d}", jahr=jahr, monat=m, tage=[d.day for d in tage]))
    return liste


def anfrage(a):
    if a["art"] == "klima":
        return MONATS_DATENSATZ, {
            "product_type": ["monthly_averaged_reanalysis"], "variable": ["sea_surface_temperature"],
            "year": [str(j) for j in range(1991, 2021)], "month": [f"{m:02d}" for m in range(1, 13)],
            "time": ["00:00"], "area": GEBIET, "data_format": "netcdf", "download_format": "unarchived"}
    return TAGES_DATENSATZ, {
        "product_type": "reanalysis", "variable": ["sea_surface_temperature"],
        "year": str(a["jahr"]), "month": [f"{a['monat']:02d}"], "day": [f"{t:02d}" for t in a["tage"]],
        "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": "1_hourly", "area": GEBIET}


def lies_offen():
    try:
        with open(OFFEN_DATEI, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def schreibe_offen(offen):
    with open(OFFEN_DATEI, "w", encoding="utf-8") as f:
        json.dump(offen, f, indent=1)


def laden(cl):
    liste = auftraege()
    offen = lies_offen()
    if not liste and not offen:
        print("Niño 3.4 (ERA5): alles vorhanden.")
        return
    print(f"Niño 3.4 (ERA5): {len(liste)} Anfrage(n) nötig, {len(offen)} warten noch beim CDS")
    reihe = lies_tage()
    beginn, pause_bis, grenze, erledigt = time.time(), 0, 1, set()
    while True:
        bleibt = []
        for a in offen:
            try:
                remote = cl.get_remote(a["id"])
                status = remote.status
            except Exception as fehler:
                if "404" in fehlertext(fehler) or "not found" in fehlertext(fehler).lower():
                    continue
                bleibt.append(a)
                continue
            if status == "successful":
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        ziel = remote.download(os.path.join(tmp, "d.nc")) or os.path.join(tmp, "d.nc")
                        s = gebietsmittel(ziel)
                    if a["art"] == "klima":
                        schreibe_klima(s)
                    else:
                        s.index = s.index.normalize()
                        reihe = pd.concat([reihe, s])
                        schreibe_tage(reihe)
                        print(f"  {a['name']}: {len(s)} Tage")
                    erledigt.add(a["name"])
                except Exception as fehler:
                    print(f"  {a['name']}: Abholen fehlgeschlagen ({fehlertext(fehler)[:300]})")
                    bleibt.append(a)
            elif status in ("failed", "rejected", "dismissed", "deleted"):
                print(f"  {a['name']}: vom CDS abgelehnt ({status}) - später erneut")
                pause_bis = time.time() + 600
                grenze = max(1, len([x for x in offen if x is not a and x.get("angenommen")]))
            else:
                if not a.get("angenommen") and (status == "running" or time.time() - a.get("seit", 0) > 300):
                    a["angenommen"] = True
                    grenze = min(MAX_OFFEN, grenze + 1)
                bleibt.append(a)
        offen = bleibt
        laufend = {a["name"] for a in offen}
        liste = [a for a in liste if a["name"] not in laufend and a["name"] not in erledigt]
        if time.time() >= pause_bis:
            while liste and len(offen) < grenze:
                a = liste.pop(0)
                try:
                    datensatz, req = anfrage(a)
                    remote = cl.submit(datensatz, req)
                    a.update(id=remote.request_id, seit=time.time())
                    offen.append(a)
                    print(f"  {a['name']}: abgeschickt")
                except Exception as fehler:
                    print(f"  {a['name']}: nicht abgeschickt ({fehlertext(fehler)[:300]})")
                    pause_bis = time.time() + 600
                    break
        schreibe_offen(offen)
        if not offen and not liste:
            print("Niño 3.4 (ERA5): alles geladen.")
            return
        if time.time() - beginn > MAX_MINUTEN * 60:
            print(f"Niño 3.4 (ERA5): Zeitlimit - {len(offen)} Anfrage(n) warten beim CDS, "
                  f"{len(liste)} noch nicht abgeschickt; weiter beim nächsten Lauf")
            return
        time.sleep(60)


if __name__ == "__main__":
    cl = client()
    if cl is None:
        print("Niño 3.4 (ERA5): kein CDS-Client gefunden - übersprungen")
    else:
        try:
            laden(cl)
        except Exception as fehler:
            print(f"Niño 3.4 (ERA5): Download nicht möglich ({type(fehler).__name__}: {fehlertext(fehler)[:500]})")
    t = lies_tage()
    print(f"Niño 3.4 (ERA5): {len(t)} Tage vorhanden, Klimamittel {'da' if os.path.exists(KLIMA_DATEI) else 'fehlt'}")
