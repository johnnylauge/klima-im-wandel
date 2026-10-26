"""
Gemeinsame Angleichung der Temperaturdatensätze an die vorindustrielle Zeit (1850-1900).

Wie Foster & Rahmstorf (GRL 2026, doi:10.1029/2025GL118804) und wie Copernicus bei ERA5:
Jeder Datensatz wird so verschoben, dass sein Mittel 1991-2020 der Erwärmung dieses Zeitraums
gegenüber 1850-1900 entspricht - im Jahresmittel 0,88 K. Wir machen das je Kalendermonat mit den
Monatswerten, die Copernicus für ERA5 nutzt (0,78-0,98 K), damit alle Datensätze auch in den
Monats- und Jahreszeitengrafiken zu ERA5 passen.

Vorteil gegenüber dem eigenen Mittel 1850-1900 jedes Datensatzes: Der Bezug liegt in einer gut
gemessenen Zeit. Die Datensätze stimmen dadurch heute überein; ihre Unterschiede zeigen sich dort,
wo die Daten unsicher sind - im 19. Jahrhundert. Ränge, Rekorde und Trends ändern sich nicht.
"""

# Erwärmung 1991-2020 gegenüber 1850-1900 je Kalendermonat (Copernicus Climate Pulse, ERA5)
OFFSETS = dict(zip(range(1, 13), [0.97, 0.98, 0.97, 0.93, 0.88, 0.83, 0.79, 0.78, 0.80, 0.83, 0.88, 0.93]))
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
