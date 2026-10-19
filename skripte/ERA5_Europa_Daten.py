"""
ERA5-Tagesmitteltemperatur für Europa (Landflächen) aus dem Copernicus Climate Data Store.

Ablauf:
  1. Lädt einmalig die Land-See-Maske von ERA5 für das Europa-Gebiet von C3S
     (34-72° N, 25° W-40° E) und merkt sie sich (ERA5_europa_lsm.nc).
  2. Lädt die Tagesmittel der 2-m-Temperatur ("ERA5 post-processed daily statistics") Jahr für
     Jahr (die neuesten Monate einzeln) und mittelt sie flächengewichtet über die Landflächen (Gitterpunkte mit mindestens 50 %
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
                       Monate (siehe EUROPA_AUFFRISCHEN) kommen immer dazu, sonst wird von 1940 an aufgefüllt
  EUROPA_MAX_MINUTEN   nach so vielen Minuten aufhören, laufende Anfragen nicht abwarten (für GitHub)
  EUROPA_AUFFRISCHEN   so viele Monate vor dem aktuellen bei jedem Lauf neu laden (Standard 3),
                       weil ERA5 dort erst vorläufig ist und später durch endgültige Werte ersetzt wird
  EUROPA_PARALLEL      gleichzeitige Anfragen an den CDS (Standard 1; der CDS begrenzt die Warteschlange)
  EUROPA_MAX_OFFEN     so viele Anfragen gleichzeitig beim CDS (Standard 6); noch wartende werden in
                       ERA5_europa_offen.json gemerkt und beim nächsten Lauf abgeholt
  EUROPA_TEIL          "alles" (Standard), "alt" (nur ab 1940 auffüllen) oder "neu" (nur die neuesten Monate)
  EUROPA_OHNE_DOWNLOAD=1  nichts laden, nur die Abweichungen neu berechnen
  EUROPA_ZUSATZ        weitere ERA5_europa_tag.txt (z. B. aus dem Nachlade-Lauf), wird zuerst eingemischt
"""

import os
import time
import zipfile
import tempfile
import threading
import sys
import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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
PARALLEL = int(os.environ.get("EUROPA_PARALLEL", "1"))
MAX_MINUTEN = float(os.environ.get("EUROPA_MAX_MINUTEN", "0")) or None
AUFFRISCHEN = int(os.environ.get("EUROPA_AUFFRISCHEN", "3"))   # so viele Monate vor dem aktuellen jedes Mal neu laden
OHNE_DOWNLOAD = os.environ.get("EUROPA_OHNE_DOWNLOAD") == "1"
BEGRENZT_WARTEN_MIN = int(os.environ.get("EUROPA_WARTEN_MIN", "120"))   # so lange warten, wenn der CDS keine Anfragen annimmt
# "alles" (Standard), "alt" = nur Lücken ab 1940 auffüllen, "neu" = nur die neuesten Monate
TEIL = os.environ.get("EUROPA_TEIL", "alles").strip().lower()

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
    return lade_zeitraum(jahr, [monat], tage, gewicht, versuche)


def lade_zeitraum(jahr, monate, tage, gewicht, versuche=3):
    """Lädt die Tagesmittel mehrerer Monate eines Jahres in EINER Anfrage (spart Wartezeit in der
    Warteschlange des CDS) und gibt das Landflächenmittel je Tag zurück (°C). Bei mehreren Monaten
    müssen alle gleich viele Tage haben (die Tagesliste gilt für jeden Monat)."""
    monat = monate[0]
    with tempfile.TemporaryDirectory() as tmp:
        ziel = os.path.join(tmp, f"t2m_{jahr}_{monat:02d}_{len(monate)}.nc")
        anfrage = {
            "product_type": "reanalysis", "variable": ["2m_temperature"],
            "year": str(jahr), "month": [f"{m:02d}" for m in monate], "day": [f"{t:02d}" for t in tage],
            "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": FREQUENZ,
            "area": GEBIET,
        }
        versuch, gewartet = 0, 0
        while True:
            versuch += 1
            try:
                client().retrieve(DATENSATZ, anfrage, ziel)
                break
            except Exception as fehler:                       # Warteschlange, Netz, Wartung ...
                if "temporarily limited" in fehlertext(fehler) and gewartet < BEGRENZT_WARTEN_MIN:
                    # Der CDS nimmt gerade keine weiteren Anfragen an (zu viele in der Warteschlange):
                    # abwarten statt aufzuteilen - mehr Anfragen würden es nur schlimmer machen
                    print(f"  {jahr}-{monat:02d}{'+' if len(monate) > 1 else ''}: CDS-Warteschlange voll - "
                          f"warte 5 Minuten ({gewartet} min bisher)")
                    time.sleep(300)
                    gewartet += 5
                    versuch -= 1                              # zählt nicht als Fehlversuch
                    continue
                if versuch >= versuche:
                    raise
                print(f"  {jahr}-{monat:02d}{'+' if len(monate) > 1 else ''}: Fehler ({fehlertext(fehler)[:400]}) - neuer Versuch in 2 Minuten")
                time.sleep(120)
        return auswerten(ziel, gewicht)


def auswerten(ziel, gewicht):
    """Landflächenmittel je Tag (°C) aus einer heruntergeladenen Datei."""
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


def fehlertext(fehler):
    """Fehlermeldung des CDS mit Begründung (steht oft nur in der Antwort des Servers)."""
    text = str(fehler)
    antwort = getattr(fehler, "response", None)
    if antwort is not None:
        try:
            text += " | " + antwort.text[:400]
        except Exception:
            pass
    return text


def tage_im_monat(jahr, m):
    return pd.Period(f"{jahr}-{m:02d}").days_in_month


def lade_jahr(jahr, monate, gewicht):
    """Die fehlenden Monate eines Jahres in wenigen Anfragen: Monate gleicher Länge zusammen
    (31 Tage, 30 Tage, Februar), damit nur gültige Tage angefragt werden. Klappt eine Gruppe
    nicht, deren Monate einzeln nacheinander."""
    gruppen = {}
    for m in monate:
        gruppen.setdefault(tage_im_monat(jahr, m), []).append(m)
    teile = []
    for laenge, ms in sorted(gruppen.items(), reverse=True):
        try:
            teile.append(lade_zeitraum(jahr, ms, list(range(1, laenge + 1)), gewicht, versuche=1))
            continue
        except Exception as fehler:
            if "temporarily limited" in fehlertext(fehler):
                raise                                  # nicht aufteilen - beim nächsten Lauf erneut
            if len(ms) == 1:
                print(f"  {jahr}-{ms[0]:02d}: nicht geladen ({fehlertext(fehler)[:600]})")
                continue
            print(f"  {jahr} ({len(ms)} Monate mit {laenge} Tagen): Anfrage nicht möglich "
                  f"({fehlertext(fehler)[:600]}) - lade sie einzeln")
        for m in ms:
            try:
                teile.append(lade_monat(jahr, m, list(range(1, laenge + 1)), gewicht))
            except Exception as f2:
                print(f"  {jahr}-{m:02d}: nicht geladen ({fehlertext(f2)[:600]})")
    if not teile:
        raise RuntimeError(f"{jahr}: nichts geladen")
    return pd.concat(teile)


# =========================================================
# 1.+2. DOWNLOAD (nur fehlende Tage)
# =========================================================
def herunterladen():
    vorhanden = lies_tage()
    letzter_tag = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=LATENZ_TAGE)
    monate = list(pd.date_range(f"{START_JAHR}-01-01", letzter_tag, freq="MS"))
    # Die neuesten Monate jedes Mal komplett neu laden: ERA5 ist dort erst vorläufig (ERA5T) und wird
    # nach etwa 2-3 Monaten durch die endgültigen Werte ersetzt - neue Werte überschreiben die alten
    neu_laden = monate[-(AUFFRISCHEN + 1):] if AUFFRISCHEN else monate[-1:]
    # Ältere Zeiträume: alle fehlenden Monate eines Jahres in EINER Anfrage. Der CDS bearbeitet
    # Anfragen nacheinander aus einer Warteschlange - je Anfrage wartet man oft viele Minuten,
    # egal wie groß sie ist. Ein Jahr auf einmal ist deshalb etwa zehnmal schneller als 12 Monate.
    jahre = {}
    for monat_start in monate:
        if monat_start in neu_laden:
            continue
        ende = monat_start + pd.offsets.MonthEnd(0)
        if any(d not in vorhanden.index for d in pd.date_range(monat_start, ende)):
            jahre.setdefault(monat_start.year, []).append(monat_start.month)
    auftraege, anzahl_monate = [], 0
    if TEIL == "neu":
        jahre = {}
    if TEIL == "alt":
        neu_laden = []
    for jahr, ms in sorted(jahre.items()):               # die ältesten fehlenden zuerst (ab 1940)
        if MAX_MONATE and anzahl_monate >= MAX_MONATE:
            break
        auftraege.append(("jahr", jahr, ms, None))
        anzahl_monate += len(ms)
    for monat_start in neu_laden:                         # neueste Monate immer dazu, einzeln
        ende = min(monat_start + pd.offsets.MonthEnd(0), letzter_tag)
        tage = list(pd.date_range(monat_start, ende))
        if AUFFRISCHEN or any(d not in vorhanden.index for d in tage):
            auftraege.append(("monat", monat_start.year, [monat_start.month], [d.day for d in tage]))
            anzahl_monate += 1
    if not auftraege:
        print("Europa: alle Tage bis", f"{letzter_tag:%d.%m.%Y}", "sind schon vorhanden.")
        return False
    print(f"Europa: {anzahl_monate} Monat(e) in {len(auftraege)} Anfragen zu laden, "
          f"{PARALLEL} Anfragen gleichzeitig ...")
    gewicht = lade_maske()
    reihe = vorhanden.copy()
    fertig = 0
    beginn = time.time()
    pool = ThreadPoolExecutor(max_workers=PARALLEL)
    neuester = auftraege[-1]
    zukunft = {}
    for a in auftraege:
        art, j, ms, t = a
        if art == "jahr":
            zukunft[pool.submit(lade_jahr, j, ms, gewicht)] = a
        else:
            zukunft[pool.submit(lade_flexibel, j, ms[0], t, gewicht, a is neuester)] = a
    offen = set(zukunft)
    abgebrochen = False
    while offen:
        erledigt, offen = wait(offen, timeout=60, return_when=FIRST_COMPLETED)
        for z in erledigt:
            art, j, ms, _ = zukunft[z]
            name = f"{j}" + (f" ({len(ms)} Monate)" if art == "jahr" else f"-{ms[0]:02d}")
            try:
                werte = z.result()
            except Exception as fehler:
                # neueste Tage oft noch nicht verfügbar -> beim nächsten Lauf erneut
                print(f"  {name}: nicht geladen ({type(fehler).__name__}: {str(fehler)[:600]})")
                continue
            with SCHLOSS:
                reihe = pd.concat([reihe, werte])
                schreibe_tage(reihe)                       # nach jeder Anfrage sichern
                fertig += 1
            vorhanden_tage = len(reihe[~reihe.index.duplicated()])
            print(f"  {name}: {len(werte)} Tage  ({fertig}/{len(auftraege)} Anfragen, "
                  f"{(time.time() - beginn) / 60:.0f} min, insgesamt {vorhanden_tage} Tage)")
        if MAX_MINUTEN and offen and time.time() - beginn > MAX_MINUTEN * 60:
            # nicht auf laufende Anfragen warten - sie können im CDS noch Stunden in der Schlange stehen
            nicht_begonnen = sum(f.cancel() for f in offen)
            print(f"Zeitlimit erreicht - {len(offen)} Anfrage(n) offen ({nicht_begonnen} noch nicht begonnen), "
                  "weiter beim nächsten Lauf")
            abgebrochen = True
            break
    pool.shutdown(wait=not abgebrochen, cancel_futures=True)
    return abgebrochen


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
        datei = os.path.join(FOLDER, f"ERA5_europa_tag_{kurz}.txt")
        soll = len(pd.date_range(f"{von}-01-01", f"{bis}-12-31"))
        ist = int(((reihe.index.year >= von) & (reihe.index.year <= bis)).sum())
        if ist < soll:                         # Bezugszeitraum noch nicht vollständig geladen
            print(f"Bezug {von}-{bis}: erst {ist} von {soll} Tagen vorhanden - noch keine Abweichungen")
            if os.path.exists(datei):
                os.remove(datei)               # keine halbfertige Datei für die Grafiken liegen lassen
            continue
        klima = klimamittel(reihe, von, bis)
        with open(datei, "w", encoding="utf-8") as f:
            f.write(f"# Tagesmitteltemperatur Europa (Landflächen), ERA5; abw = Abweichung zum Mittel {von}-{bis}\n")
            f.write("date,2t,abw\n")
            for d, v in reihe.items():
                f.write(f"{d:%Y-%m-%d},{v:.4f},{v - klima[(d.month, d.day)]:.4f}\n")
        print(f"Abweichungen zu {von}-{bis} gespeichert unter:\n{datei}")


# =========================================================
# 1.+2. DOWNLOAD MIT MERKZETTEL (Anfragen abschicken, später abholen)
# =========================================================
# Der CDS ist oft stark ausgelastet: Anfragen warten dann viele Stunden in der Schlange - länger als
# ein Lauf auf GitHub dauern darf. Deshalb werden Anfragen nur abgeschickt und ihre Nummern in
# ERA5_europa_offen.json gemerkt. Jeder Lauf schaut nach, welche fertig sind, holt sie ab und schickt
# neue nach. So geht kein Platz in der Warteschlange verloren, wenn ein Lauf endet.
OFFEN_DATEI = os.path.join(FOLDER, "ERA5_europa_offen.json")
MAX_OFFEN = int(os.environ.get("EUROPA_MAX_OFFEN", "6"))     # so viele Anfragen gleichzeitig beim CDS
JE_ANFRAGE = os.environ.get("EUROPA_JE_ANFRAGE", "monat").strip().lower()   # "monat" oder "jahr"


def asynchron_client():
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


def anfrage_fuer(a):
    return {
        "product_type": "reanalysis", "variable": ["2m_temperature"],
        "year": str(a["jahr"]), "month": [f"{m:02d}" for m in a["monate"]],
        "day": [f"{t:02d}" for t in a["tage"]],
        "daily_statistic": "daily_mean", "time_zone": "utc+00:00", "frequency": FREQUENZ,
        "area": GEBIET,
    }


def name_von(a):
    return f"{a['jahr']}-{a['monate'][0]:02d}" if len(a["monate"]) == 1 else f"{a['jahr']} ({len(a['monate'])} Monate)"


def schluessel(a):
    return f"{a['jahr']}:{','.join(str(m) for m in a['monate'])}"


def auftraege_bauen(vorhanden):
    """Was fehlt? Ältere Jahre: alle fehlenden Monate eines Jahres in einer Anfrage (Tage 1-31,
    zusammenhängend - der CDS ist bei zusammenhängenden Tagen deutlich schneller). Neueste Monate
    (vorläufige ERA5T-Werte) einzeln und jedes Mal neu."""
    letzter_tag = pd.Timestamp.now("UTC").tz_localize(None).normalize() - pd.Timedelta(days=LATENZ_TAGE)
    monate = list(pd.date_range(f"{START_JAHR}-01-01", letzter_tag, freq="MS"))
    neu_laden = monate[-(AUFFRISCHEN + 1):] if AUFFRISCHEN else monate[-1:]
    jahre = {}
    for monat_start in monate:
        if monat_start in neu_laden:
            continue
        ende = monat_start + pd.offsets.MonthEnd(0)
        if any(d not in vorhanden.index for d in pd.date_range(monat_start, ende)):
            jahre.setdefault(monat_start.year, []).append(monat_start.month)
    if TEIL == "neu":
        jahre = {}
    if TEIL == "alt":
        neu_laden = []
    auftraege, anzahl = [], 0
    for jahr, ms in sorted(jahre.items()):               # die ältesten fehlenden zuerst (ab 1940)
        if MAX_MONATE and anzahl >= MAX_MONATE:
            break
        if JE_ANFRAGE == "jahr":
            auftraege.append(dict(jahr=jahr, monate=ms, tage=list(range(1, 32))))
        else:
            # je Monat eine Anfrage: Anfragen über mehr als einen Monat zählen beim CDS als "groß"
            # und haben eine eigene, viel vollere Warteschlange
            auftraege += [dict(jahr=jahr, monate=[m],
                               tage=list(range(1, pd.Period(f"{jahr}-{m:02d}").days_in_month + 1))) for m in ms]
        anzahl += len(ms)
    for monat_start in neu_laden:
        ende = min(monat_start + pd.offsets.MonthEnd(0), letzter_tag)
        tage = [d.day for d in pd.date_range(monat_start, ende)]
        auftraege.append(dict(jahr=monat_start.year, monate=[monat_start.month], tage=tage, neu=True))
    return auftraege, letzter_tag


def lies_offen():
    try:
        with open(OFFEN_DATEI, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def schreibe_offen(offen):
    with open(OFFEN_DATEI, "w", encoding="utf-8") as f:
        json.dump(offen, f, indent=1)


def herunterladen_merkzettel(cl):
    vorhanden = lies_tage()
    auftraege, letzter_tag = auftraege_bauen(vorhanden)
    gemerkt = lies_offen()
    offen = [a for a in gemerkt if a.get("id")]
    # Monate abgelehnter Jahresanfragen, die einzeln geladen werden sollen (ohne Nummer gemerkt)
    einzeln = [a for a in gemerkt if not a.get("id")]
    geteilt = {a["jahr"] for a in einzeln}
    auftraege = einzeln + [a for a in auftraege if a.get("neu") or a["jahr"] not in geteilt]
    if not auftraege and not offen:
        print("Europa: alle Tage bis", f"{letzter_tag:%d.%m.%Y}", "sind schon vorhanden.")
        return
    print(f"Europa: {len(auftraege)} Anfrage(n) nötig, {len(offen)} vom letzten Lauf noch beim CDS; "
          f"höchstens {MAX_OFFEN} gleichzeitig (Start mit 1, steigt, solange der CDS annimmt)")
    gewicht = None
    reihe = vorhanden.copy()
    beginn = time.time()
    pause_bis = 0
    letzte_meldung = 0
    erledigt = set()                                         # in diesem Lauf schon abgeholt
    # Wie viele Anfragen der CDS gerade annimmt, ist nicht bekannt und schwankt mit der Auslastung:
    # vorsichtig mit 1 anfangen, nach jeder angenommenen Anfrage eine mehr erlauben (bis MAX_OFFEN),
    # nach einer Ablehnung zurück auf die Zahl, die gerade wartet (mindestens 1)
    grenze = 1
    while True:
        # 1. abgeschickte Anfragen prüfen und fertige abholen
        bleibt = []
        for a in offen:
            try:
                remote = cl.get_remote(a["id"])
                status = remote.status
            except Exception as fehler:
                print(f"  {name_von(a)}: Status nicht abrufbar ({fehlertext(fehler)[:300]})")
                if "not found" in fehlertext(fehler).lower() or "404" in fehlertext(fehler):
                    continue                                 # beim CDS gelöscht -> neu abschicken
                bleibt.append(a)
                continue
            if status == "successful":
                try:
                    if gewicht is None:
                        gewicht = lade_maske()
                    with tempfile.TemporaryDirectory() as tmp:
                        ziel = remote.download(os.path.join(tmp, "daten.nc")) or os.path.join(tmp, "daten.nc")
                        werte = auswerten(ziel, gewicht)
                    reihe = pd.concat([reihe, werte])
                    schreibe_tage(reihe)
                    erledigt.add(schluessel(a))
                    gesamt = len(reihe[~reihe.index.duplicated()])
                    print(f"  {name_von(a)}: {len(werte)} Tage  ({(time.time() - beginn) / 60:.0f} min, "
                          f"insgesamt {gesamt} Tage)")
                except Exception as fehler:
                    print(f"  {name_von(a)}: Abholen fehlgeschlagen ({fehlertext(fehler)[:400]})")
                    bleibt.append(a)
            elif status in ("failed", "rejected", "dismissed", "deleted"):
                grund = ""
                try:
                    grund = str(getattr(remote, "json", "") or "")[:300]
                except Exception:
                    pass
                print(f"  {name_von(a)}: vom CDS abgelehnt ({status}) {grund}")
                if status == "rejected" or "limited" in grund:
                    pause_bis = time.time() + 600                # Schlange voll: 10 min nichts Neues schicken
                    grenze = max(1, len([x for x in offen if x is not a and x.get("angenommen")]))
                    if a.get("neu") or len(a["monate"]) == 1:
                        auftraege.insert(0, {k: v for k, v in a.items() if k != "id"})   # später erneut
                elif len(a["monate"]) > 1:
                    # z. B. ungültige Tage: Monate einzeln mit gültigen Tagen nachschieben
                    auftraege[:0] = [dict(jahr=a["jahr"], monate=[m], einzeln=True,
                                          tage=list(range(1, pd.Period(f"{a['jahr']}-{m:02d}").days_in_month + 1)))
                                     for m in a["monate"]]
            else:
                if not a.get("angenommen") and (status == "running" or time.time() - a.get("seit", 0) > 300):
                    a["angenommen"] = True
                    grenze = min(MAX_OFFEN, grenze + 1)      # der CDS nimmt an - eine mehr erlauben
                bleibt.append(a)                             # accepted / running: weiter warten
        offen = bleibt
        # 2. neue Anfragen abschicken, solange Platz ist
        laufend = {schluessel(a) for a in offen}
        auftraege = [a for a in auftraege if schluessel(a) not in laufend and schluessel(a) not in erledigt]
        if time.time() >= pause_bis:
            while auftraege and len(offen) < grenze:
                a = auftraege.pop(0)
                if not a.get("neu"):
                    # schon vorhanden (z. B. gerade abgeholt)?
                    tage = [d for m in a["monate"] for d in pd.date_range(
                        f"{a['jahr']}-{m:02d}-01", pd.Timestamp(f"{a['jahr']}-{m:02d}-01") + pd.offsets.MonthEnd(0))]
                    if all(d in reihe.index for d in tage):
                        continue
                try:
                    remote = cl.submit(DATENSATZ, anfrage_fuer(a))
                    a["id"] = remote.request_id
                    a["seit"] = time.time()
                    offen.append(a)
                    print(f"  {name_von(a)}: abgeschickt")
                except Exception as fehler:
                    print(f"  {name_von(a)}: nicht abgeschickt ({fehlertext(fehler)[:300]})")
                    auftraege.insert(0, a)
                    pause_bis = time.time() + 600
                    break
        # merken: abgeschickte Anfragen und schon aufgeteilte Monate
        schreibe_offen(offen + [a for a in auftraege if a.get("einzeln")])
        if not offen and not auftraege:
            print("Europa: alles geladen.")
            return
        if MAX_MINUTEN and time.time() - beginn > MAX_MINUTEN * 60:
            print(f"Zeitlimit erreicht - {len(offen)} Anfrage(n) warten beim CDS, {len(auftraege)} noch nicht "
                  "abgeschickt; der nächste Lauf holt sie ab")
            return
        if time.time() - letzte_meldung > 1800:              # alle 30 min ein Lebenszeichen
            print(f"  ... {len(offen)} Anfrage(n) warten beim CDS ({(time.time() - beginn) / 60:.0f} min)")
            letzte_meldung = time.time()
        time.sleep(60)


# Tage aus einer zweiten Datei übernehmen (der Nachlade-Lauf speichert seinen Stand getrennt)
ZUSATZ = os.environ.get("EUROPA_ZUSATZ", "")
if ZUSATZ and os.path.exists(ZUSATZ) and os.path.abspath(ZUSATZ) != os.path.abspath(TAGE_DATEI):
    alt = lies_tage()
    neu = pd.read_csv(ZUSATZ, comment="#", parse_dates=["date"]).set_index("date")["2t"]
    zusammen = pd.concat([neu, alt])                      # vorhandene (alt) haben Vorrang
    zusammen = zusammen[~zusammen.index.duplicated(keep="last")]
    if len(zusammen) > len(alt):
        schreibe_tage(zusammen)
        print(f"Aus {ZUSATZ} übernommen: {len(zusammen) - len(alt)} Tage")

abgebrochen = False
if not OHNE_DOWNLOAD:
    try:
        cl = asynchron_client()
        if cl is not None:
            herunterladen_merkzettel(cl)
        else:                        # alter Weg: warten, bis jede Anfrage fertig ist
            print("Hinweis: kein Client zum späteren Abholen gefunden - lade blockierend")
            abgebrochen = herunterladen()
    except Exception as fehler:      # z. B. CDS nicht erreichbar oder kein Schlüssel: mit den vorhandenen Tagen weiter
        print(f"Download nicht möglich: {type(fehler).__name__}: {str(fehler)[:1500]}")
        print("Rechne mit den vorhandenen Tageswerten weiter.")
abweichungen()
if abgebrochen:
    # noch laufende Anfragen nicht abwarten (Python würde sonst beim Beenden auf sie warten)
    sys.stdout.flush()
    os._exit(0)
