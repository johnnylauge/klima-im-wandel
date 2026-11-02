"""
Gemeinsame Angleichung der Temperaturdatensätze an die vorindustrielle Zeit (1850-1900).

Wie Foster & Rahmstorf (GRL 2026, doi:10.1029/2025GL118804) und wie Copernicus bei ERA5:
Jeder Datensatz wird so verschoben, dass sein Mittel 1991-2020 der Erwärmung dieses Zeitraums
gegenüber 1850-1900 entspricht - im Jahresmittel 0,88 K. Wir machen das je Kalendermonat mit den
Monatswerten, die Copernicus für ERA5 nutzt (0,80-0,96 K), damit alle Datensätze auch in den
Monats- und Jahreszeitengrafiken zu ERA5 passen.

ERA5-Tageswerte bekommen wie bei Copernicus einen Abstand je Tag (offset_tag), siehe
https://climate.copernicus.eu/climate-bulletin-about-data-and-analysis
(Fassung seit Dezember 2023; die frühere vorläufige Fassung von Juni 2023 wich um bis zu 0,02 K ab).

Vorteil gegenüber dem eigenen Mittel 1850-1900 jedes Datensatzes: Der Bezug liegt in einer gut
gemessenen Zeit. Die Datensätze stimmen dadurch heute überein; ihre Unterschiede zeigen sich dort,
wo die Daten unsicher sind - im 19. Jahrhundert. Ränge, Rekorde und Trends ändern sich nicht.
"""

import numpy as np
import pandas as pd

# Erwärmung 1991-2020 gegenüber 1850-1900 je Kalendermonat ("Applied Offset" von Copernicus,
# Monatsmittel der Tagesformel unten)
OFFSETS = dict(zip(range(1, 13), [0.96, 0.96, 0.95, 0.91, 0.87, 0.83, 0.80, 0.80, 0.81, 0.85, 0.89, 0.93]))


def offset_tag(daten, offsets=None):
    """Abstand 1991-2020 -> 1850-1900 für jeden Tag, wie Copernicus:
    0,88 + 0,05·sin(2π(n−0,5)/365) + 0,07·cos(2π(n−0,5)/365), n = Tag im 365-Tage-Jahr;
    der 29. Februar liegt zwischen 28. Februar und 1. März.
    offsets: das Monats-Wörterbuch des Skripts - sind dort alle Werte 0 (Europa), ist der Abstand 0."""
    d = pd.to_datetime(pd.Series(daten))
    if offsets is not None and not any(offsets.values()):
        return pd.Series(0.0, index=d.index)
    tag = d.dt.dayofyear.astype(float)
    schalt = d.dt.is_leap_year
    n = tag - (schalt & (tag > 60)).astype(float)
    n = n.where(~(schalt & (tag == 60)), 59.5)
    x = 2 * np.pi * (n - 0.5) / 365
    return 0.88 + 0.05 * np.sin(x) + 0.07 * np.cos(x)
VON, BIS = 1991, 2020
HINWEIS = "angeglichen über 1991-2020"


def auf_1850_1900(df, spalte):
    """Abweichung zu 1850-1900: Abweichung zum eigenen Mittel 1991-2020 (je Kalendermonat)
    plus der Copernicus-Erwärmung dieses Monats. df braucht die Spalten 'jahr', 'monat', spalte."""
    basis = df[(df["jahr"] >= VON) & (df["jahr"] <= BIS)]
    if basis.groupby("monat")["jahr"].nunique().min() < BIS - VON + 1:
        raise ValueError(f"Zeitraum {VON}-{BIS} nicht vollständig - Angleichung nicht möglich")
    mittel = basis.groupby("monat")[spalte].mean()
    return df[spalte] - df["monat"].map(mittel) + df["monat"].map(OFFSETS)
