"""
Gemeinsame Einstellungen für die Region der ERA5-Grafiken (global oder Europa).

Die ERA5-Skripte lesen hier, für welches Gebiet sie rechnen. Gesteuert wird das über
Umgebungsvariablen (auf GitHub in der Steuerdatei, lokal z. B. per "set ERA5_REGION=europa"):

  ERA5_REGION   "global" (Standard) oder "europa"
  ERA5_BEZUG    nur bei Europa: "1961-1990" (Standard) oder "1991-2020"

Global:  Tageswerte von Copernicus Climate Pulse, Abweichung bezogen auf 1850-1900.
Europa:  Tagesmittel aus dem Copernicus Climate Data Store (ERA5 post-processed daily statistics),
         über die Landflächen 34-72° N, 25° W-40° E gemittelt (Europa-Gebiet von C3S);
         ERA5_Europa_Daten.py lädt und berechnet sie. Abweichung bezogen auf 1961-1990 oder
         1991-2020 - ERA5 beginnt 1940, der Bezug 1850-1900 ist hier nicht möglich.
"""

import os


def dz(v, fmt=".2f"):
    """Zahl in deutscher Schreibweise für Grafiktexte: Komma statt Punkt, echtes Minus."""
    return format(v, fmt).replace(".", ",").replace("-", "−")

REGION = os.environ.get("ERA5_REGION", "global").strip().lower()
if REGION not in ("global", "europa"):
    raise ValueError(f"ERA5_REGION '{REGION}' unbekannt - möglich: global, europa")
EUROPA = REGION == "europa"

if EUROPA:
    BEZUG = os.environ.get("ERA5_BEZUG", "1961-1990").strip()
    if BEZUG not in ("1961-1990", "1991-2020"):
        raise ValueError(f"ERA5_BEZUG '{BEZUG}' unbekannt - möglich: 1961-1990, 1991-2020")
    KURZ = BEZUG[2:4] + BEZUG[7:9]                     # "6190" oder "9120"
    GEBIET = "Europa"
    STAMM = f"ERA5_europa_{KURZ}"                      # Dateinamen, z. B. ERA5_europa_6190_jahreswerte
    TAG_DATEI = f"ERA5_europa_tag_{KURZ}.txt"          # Tageswerte mit Abweichung zu diesem Bezug
    DATENNAME = "ERA5, Copernicus Climate Data Store, Landflächen"
    UNTERTITEL = f"Abweichung vom Klimamittel {BEZUG} · Landflächen"
    FUSS = ("Datenquelle: ERA5, C3S/ECMWF (Climate Data Store) · CC BY 4.0 · eigene Berechnung\n"
            "Europa: Landflächen 34–72° N, 25° W–40° E")
else:
    BEZUG = "1850-1900"
    KURZ = ""
    GEBIET = "Global"
    STAMM = "ERA5_global"
    TAG_DATEI = "ERA5_global_tag.txt"
    DATENNAME = "ERA5, Copernicus Climate Pulse"
    UNTERTITEL = "Abweichung von der vorindustriellen Zeit (1850-1900)"
    FUSS = ("Datenquelle: Copernicus Climate Pulse (ERA5), C3S/ECMWF · CC BY 4.0\n"
            "https://pulse.climate.copernicus.eu/ · eigene Berechnung")

# 1,5-K-Grenze nur beim vorindustriellen Bezug sinnvoll
MIT_1_5 = not EUROPA


def anpassen_tage(df):
    """Europa: Spalte 'abw' (Abweichung zum gewählten Bezug) unter dem Namen der globalen
    Spalte bereitstellen, damit die Skripte unverändert weiterrechnen."""
    if EUROPA and "abw" in df.columns:
        df = df.rename(columns={"abw": "ano_91-20"})
    return df


def temp_info(tabelle, jahr, unvollstaendig):
    """Temperatur für den Info-Kasten. Europa, laufender Zeitraum: Der bisherige Mittelwert hängt
    stark von der Jahreszeit ab (Jan.-Okt. ist viel wärmer als das ganze Jahr). Deshalb wird er
    über die bisherige Abweichung aufs ganze Jahr hochgerechnet: Abweichung + mittlerer Abstand
    Temperatur - Abweichung der vollständigen Zeiträume."""
    z = tabelle.loc[jahr]
    if not (EUROPA and unvollstaendig):
        return f"{dz(z['temp'], '.2f')} °C"
    voll = tabelle.drop(jahr)
    abstand = float((voll["temp"] - voll["anomaly"]).median())
    return f"≈ {dz(z['anomaly'] + abstand, '.2f')} °C (hochgerechnet)"


def temp_hover(tabelle, jahr, laufend_jahr):
    """Temperatur für die Maus-Anzeige der Webseite; Europa, laufender Zeitraum: hochgerechnet
    (ca=True, die Webseite zeigt dann "≈")."""
    z = tabelle.loc[jahr]
    if EUROPA and laufend_jahr is not None and jahr == laufend_jahr:
        voll = tabelle.drop(jahr)
        abstand = float((voll["temp"] - voll["anomaly"]).median())
        return dict(temp=round(float(z["anomaly"] + abstand), 3), ca=True)
    return dict(temp=round(float(z["temp"]), 3))
