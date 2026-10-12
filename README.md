# Klimagrafiken – Globale Jahresmitteltemperatur (ERA5)

Täglich um 09:00 UTC lädt GitHub die aktuellen ERA5-Tageswerte von
[Copernicus Climate Pulse](https://pulse.climate.copernicus.eu/), erstellt daraus die Grafik
der globalen Jahresmitteltemperatur (Abweichung von 1850–1900) und veröffentlicht sie auf
GitHub Pages.

## Aufbau

| Datei / Ordner | Zweck |
| --- | --- |
| `skripte/ERA5_Jahresmittel_Balken.py` | Lädt die Daten, rechnet die Jahreswerte, zeichnet die Grafik |
| `site/index.html` | Die Webseite, auf der Grafik, Kennzahlen und Downloads erscheinen |
| `.github/workflows/taeglich.yml` | Steuerdatei: wann und wie GitHub das Skript ausführt |
| `requirements.txt` | Python-Pakete, die GitHub installiert |

Die erzeugten Dateien (PNG, TXT) landen nicht im Repository, sondern direkt auf der Webseite.
So wächst das Repository nicht mit jedem Tag.

## Lokal ausführen

Das Skript läuft weiterhin unverändert auf Windows: Ohne die Umgebungsvariable `ERA5_FOLDER`
speichert es in `C:\Users\rafff\Documents\Wetterdaten\ERA5\Climate_Pulse` und öffnet das
Grafikfenster.

## Einrichtung (einmalig)

1. Repository auf GitHub anlegen (öffentlich), alle Dateien hochladen.
2. **Settings → Pages → Build and deployment → Source: „GitHub Actions“** wählen.
3. **Actions → Tägliche Klimagrafik → Run workflow** einmal von Hand starten.
4. Nach 2–3 Minuten ist die Seite erreichbar unter `https://<benutzername>.github.io/<repository>/`.

## Hinweise

- GitHub startet geplante Läufe zur vollen Stunde oft mit Verzögerung (meist einige Minuten,
  bei hoher Last auch länger).
- Bei öffentlichen Repositories pausiert GitHub geplante Läufe, wenn 60 Tage lang nichts am
  Repository geändert wurde. GitHub schickt dann eine E-Mail; unter **Actions** lässt sich der
  Workflow mit einem Klick wieder aktivieren.
