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
raster-impress dop20rgb_33316_5690_2_sn.tif --stats --histogram --ndvi --slope --hillshade --metadata --quality

# Analyse von Slope und NDVI mit und ohne Speicherpfad
raster-impress dop20rgb_33316_5690_2_sn.tif --slope output/myslope.tif --ndvi output/myndvi.tif
raster-impress dop20rgb_33316_5690_2_sn.tif --slope --ndvi



```

## Lizenz

Projekt kann unter MIT-Lizenz oder Open-Source Lizenz genutzt werden.
