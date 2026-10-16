"""
Bereitet die Daten-Downloads der Webseite einheitlich auf:

  1. ERA5_global_tageswerte.txt: alle Tageswerte von ERA5 im gleichen Stil wie die anderen
     TXT-Dateien (Datum, Temperatur in °C, Abweichung von 1850-1900).
  2. Zu jeder TXT-Datei *_global_*.txt (Jahres-, Jahreszeiten-, Monatswerte, Monatsreihen,
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
if os.path.exists(ERA5_ROH):
    tage = pd.read_csv(ERA5_ROH, comment="#", parse_dates=["date"])
    tage["abw"] = tage["ano_91-20"] + tage["date"].dt.month.map(MONTHLY_OFFSETS)
    with open(ERA5_TAGE, "w", encoding="utf-8") as f:
        f.write("# Tagesmitteltemperatur – Global (ERA5, Copernicus Climate Pulse)\n"
                "# Abweichung bezogen auf 1850-1900\n"
                f"# Stand: {tage['date'].max():%d.%m.%Y}\n"
                "#\n"
                f"{'Datum':>10}  {'Temp_°C':>8}  {'Abw_K':>6}\n")
        for d, t, a in zip(tage["date"], tage["2t"], tage["abw"]):
            f.write(f"{d:%d.%m.%Y}  {t:>8.3f}  {a:>+6.3f}\n")
    print(f"Tageswerte gespeichert unter:\n{ERA5_TAGE}")
else:
    print(f"Hinweis: {ERA5_ROH} fehlt - keine ERA5-Tageswerte")


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
for txt in sorted(glob.glob(os.path.join(FOLDER, "*_global_*.txt"))):
    if os.path.abspath(txt) == os.path.abspath(ERA5_ROH):
        continue
    anzahl += als_csv(txt)
print(f"{anzahl} CSV-Dateien geschrieben in:\n{FOLDER}")
