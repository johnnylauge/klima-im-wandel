"""
Bereitet die Daten-Downloads der Webseite einheitlich auf:

  1. ERA5-Tageswerte im gleichen Stil wie die anderen TXT-Dateien (Datum, Temperatur in °C,
     Abweichung von 1850-1900) - passend zu den Punktdiagrammen: alle Tage
     (ERA5_global_tageswerte.txt), je Jahreszeit (..._saison_herbst.txt) und je Monat (..._monat_10.txt).
     Für Europa genauso je Bezugszeitraum (ERA5_europa_6190_tageswerte..., ERA5_europa_9120_...).
  2. Zu jeder TXT-Datei *_global_*.txt und ERA5_europa_*.txt (Jahres-, Jahreszeiten-, Monatswerte, Tageswerte,
     Vergleichstabellen) eine CSV-Datei für Excel: Trennzeichen Semikolon, Dezimalkomma,
     UTF-8 mit BOM (damit Excel °C und Umlaute richtig zeigt). Die Kopfzeilen mit # (Quelle,
     Stand, Bezug) stehen als Text in den ersten Zeilen.

Lädt nichts herunter; läuft auf GitHub als letzter Schritt, nachdem alle Grafiken erstellt sind.
  ERA5_FOLDER  Datenordner (dort liegen alle TXT-Dateien)
"""

import os
import re
import glob
import pandas as pd

FOLDER = os.environ.get("ERA5_FOLDER", r"C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse")
ERA5_ROH = os.path.join(FOLDER, "ERA5_global_tag.txt")          # Original von Climate Pulse
ERA5_TAGE = os.path.join(FOLDER, "ERA5_global_tageswerte.txt")

# Vorindustrielle Offsets je Monat (1991-2020 -> 1850-1900), wie in den ERA5-Skripten
MONTHLY_OFFSETS = dict(zip(range(1, 13), [
    0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93,
]))

ZAHL = re.compile(r"^[+-]?\d+\.\d+$")


# =========================================================
# 1. ERA5-Tageswerte im einheitlichen Stil
# =========================================================
MONATE = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli",
          "August", "September", "Oktober", "November", "Dezember"]
JAHRESZEITEN = {"fruehling": ("Frühling", [3, 4, 5]), "sommer": ("Sommer", [6, 7, 8]),
                "herbst": ("Herbst", [9, 10, 11]), "winter": ("Winter", [12, 1, 2])}


def schreibe_tage(daten, datei, titel, stand, gebiet="Global (ERA5, Copernicus Climate Pulse)", bezug="1850-1900"):
    with open(datei, "w", encoding="utf-8") as f:
        f.write(f"# Tagesmitteltemperatur{titel} – {gebiet}\n"
                f"# Abweichung bezogen auf {bezug}\n"
                f"# Stand: {stand:%d.%m.%Y}\n"
                "#\n"
                f"{'Datum':>10}  {'Temp_°C':>8}  {'Abw_K':>6}\n")
        for d, t, a in zip(daten["date"], daten["2t"], daten["abw"]):
            f.write(f"{d:%d.%m.%Y}  {t:>8.3f}  {a:>+6.3f}\n")


def tage_je_zeitraum(tage, basis, gebiet, bezug):
    """Alle Tage (zu den Jahres-Punktdiagrammen) sowie nur die Tage je Jahreszeit und je Monat."""
    stand = tage["date"].max()
    schreibe_tage(tage, basis + ".txt", "", stand, gebiet, bezug)
    for datei, (name, monate) in JAHRESZEITEN.items():
        schreibe_tage(tage[tage["date"].dt.month.isin(monate)], f"{basis}_saison_{datei}.txt", f" {name}",
                      stand, gebiet, bezug)
    for m, name in enumerate(MONATE, start=1):
        schreibe_tage(tage[tage["date"].dt.month == m], f"{basis}_monat_{m:02d}.txt", f" {name}",
                      stand, gebiet, bezug)
    print(f"Tageswerte {os.path.basename(basis)} (gesamt, je Jahreszeit, je Monat) gespeichert")


# Global: Climate Pulse, Abweichung zu 1991-2020 plus Offset -> 1850-1900
if os.path.exists(ERA5_ROH):
    tage = pd.read_csv(ERA5_ROH, comment="#", parse_dates=["date"])
    tage["abw"] = tage["ano_91-20"] + tage["date"].dt.month.map(MONTHLY_OFFSETS)
    tage_je_zeitraum(tage, os.path.splitext(ERA5_TAGE)[0], "Global (ERA5, Copernicus Climate Pulse)", "1850-1900")
else:
    print(f"Hinweis: {ERA5_ROH} fehlt - keine globalen ERA5-Tageswerte")

# Europa: je Bezugszeitraum (von ERA5_Europa_Daten.py berechnet)
for kurz, bezug in (("6190", "1961-1990"), ("9120", "1991-2020")):
    quelle = os.path.join(FOLDER, f"ERA5_europa_tag_{kurz}.txt")
    if os.path.exists(quelle):
        tage = pd.read_csv(quelle, comment="#", parse_dates=["date"])
        tage_je_zeitraum(tage, os.path.join(FOLDER, f"ERA5_europa_{kurz}_tageswerte"),
                         "Europa (ERA5, Copernicus Climate Data Store, Landflächen)", bezug)


# =========================================================
# 2. CSV für Excel zu jeder TXT-Datei
# =========================================================
def als_csv(txt):
    zeilen = open(txt, encoding="utf-8").read().splitlines()
    if not zeilen or not zeilen[0].startswith("# "):
        return False                                    # keine eigene TXT (z. B. Originaldatei)
    aus = []
    for z in zeilen:
        if z.startswith("#"):
            text = z.lstrip("#").strip()
            aus.append('"' + text.replace('"', '""') + '"' if text else "")
        elif z.strip():
            teile = z.split()
            aus.append(";".join(t.lstrip("+").replace(".", ",") if ZAHL.match(t) else t for t in teile))
    csv = os.path.splitext(txt)[0] + ".csv"
    with open(csv, "w", encoding="utf-8-sig", newline="") as f:
        f.write("\r\n".join(aus) + "\r\n")
    return True


anzahl = 0
ROHDATEIEN = {"ERA5_global_tag.txt", "ERA5_europa_tag.txt", "ERA5_europa_tag_6190.txt", "ERA5_europa_tag_9120.txt"}
for txt in sorted(glob.glob(os.path.join(FOLDER, "*_global_*.txt")) + glob.glob(os.path.join(FOLDER, "ERA5_europa_*.txt"))):
    if os.path.basename(txt) in ROHDATEIEN:              # Rohdaten (CSV mit Komma), keine eigene Tabelle
        continue
    anzahl += als_csv(txt)
print(f"{anzahl} CSV-Dateien geschrieben in:\n{FOLDER}")
