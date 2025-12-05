# Raster Impress CLI

**Raster Impress CLI** ist ein Kommandozeilen-Tool für die Analyse von Rasterdaten (GeoTIFF, NetCDF, HDF5) unter Linux/macOS.
Es liefert **Statistiken, Histogramme, NDVI, Slope, Hillshade, Zonal Stats und Visualisierung** und lässt sich direkt als CLI verwenden.

---

## Repository klonen

```bash
git clone https://github.com/deinuser/raster-impress-cli.git
cd raster-impress-cli
```

## Installation der Abhängigkeiten

```bash
pip install -r requirements.txt
python -m pip install --upgrade pip setuptools wheel
```

## Installation als Linux CLI

```bash
sudo apt install pipx
pipx ensurepath
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
# oder systemweit mit pipx
pipx install --editable . --include-deps
```

```bash
raster-impress --version
```

* Prüfe, dass `~/.local/bin` in deinem `$PATH` ist:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

* CLI aufrufen:

```bash
man raster-impress
```

```bash
raster-impress --version
```

```bash
raster-impress --help
```

## Rasterdaten ablegen

```bash
mkdir -p ~/raster_data
cp my_raster.tif ~/raster_data/
```

## Manpage installieren (Linux)

```bash
sudo cp docs/raster-impress.1 /usr/share/man/man1/
sudo mandb
man raster-impress
```

## Tests ausführen

```bash
pip install pytest
pytest tests -v
```

## CLI-Parameter

```bash
raster-impress --help

usage: raster-impress [-h] [--stats] [--histogram [HISTOGRAM]] [--ndvi [NDVI]] [--slope [SLOPE]] [--hillshade [HILLSHADE]]
                      [--relief [RELIEF]] [--metadata] [--quality] [--silent] [--version]
                      filepath

Raster analysis tool with automatic TIF and plot generation

positional arguments:
  filepath              Path to input raster file

options:
  -h, --help            show this help message and exit
  --stats               Compute basic statistics
  --histogram [HISTOGRAM]
                        Compute histogram. Optional output filename
  --ndvi [NDVI]         Compute NDVI (requires at least 2 bands). Optional output filename
  --slope [SLOPE]       Compute Slope (DEM required). Optional output filename
  --hillshade [HILLSHADE]
                        Compute Hillshade (DEM required). Optional output filename
  --relief [RELIEF]     Compute synthetic Relief (DEM required). Optional output filename
  --metadata            Show raster metadata
  --quality             Perform quality check
  --silent              Suppress log output
  --version             show program's version number and exit
```

## CI/CD Pipeline (GitHub Actions)

* `.github/workflows/python-package.yml`
* Repository auschecken
* Python >= 3.11 installieren
* Abhängigkeiten installieren
* Tests ausführen
* Linting mit flake8
* Push/PR auf `main` löst automatische Ausführung aus


## Zusammenfassung der Befehle

```bash

# Vollanalyse mit allen Funktionen ohne Speicherpfad
raster-impress raster.tif --stats --histogram --ndvi --slope --hillshade --metadata --quality

# Einzelanalyse
raster-impress dem.tif --slope
raster-impress dem.tif --hillshade
raster-impress dem.tif --relief

raster-impress raster.tif --ndvi
```

## Beispiele

| Beschriftung | Bild                                       |
|--------------|--------------------------------------------|
| Slope        | ![Screenshot slope](examples/slope.png)    |
| Hillshade    | ![Screenshot hill](examples/hillshade.png) |
| Relief       | ![Screenshot relief](examples/relief.png)  |
| NDVI         | ![Screenshot ndvi](examples/ndvi.png)      |

## Datenquelle

Offene Geodaten des Freistaates Sachsen:
[Downloadbereich Offene Geodaten](https://www.geodaten.sachsen.de/index.html)

Lizenz: Datenlizenz Deutschland – Namensnennung – Version 2.0  
© Staatsbetrieb Geobasisinformation und Vermessung Sachsen (GeoSN)


## Lizenz

Projekt kann unter MIT-Lizenz oder Open-Source Lizenz genutzt werden.
