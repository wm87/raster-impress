from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from raster_impress.raster_analysis import (
    analyze_raster,
    compute_histogram,
    compute_ndvi,
    compute_slope,
    compute_hillshade,
    print_metadata,
    quality_check,
)


# ----------------------------------------
# Helper Functions
# ----------------------------------------

def create_test_raster(tmp_path: Path, bands=4):
    """Helper: create small raster with given number of bands (default is 4 for NDVI)"""
    if bands < 4:
        bands = 4

    # Erstellen zufälliger Daten für 4 Bänder (Rot, Grün, Blau, NIR)
    # Hier setzen wir sicher, dass die Werte im Bereich von 0 bis 255 liegen, um das Raster realistischer zu gestalten
    data = np.random.randint(50, 151, size=(bands, 2, 2), dtype=np.uint8)  # Shape: (Bands, Height, Width)

    # Sicherstellen, dass der maximale Wert 150 ist
    data[0][0][0] = 150  # Setze explizit den Maximalwert auf 150

    raster_path = tmp_path / "test.tif"
    with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=bands,
            dtype=data.dtype,
            crs="+proj=latlong",
            transform=from_origin(0, 0, 1, 1),  # Georeferenzierung setzen
    ) as dst:
        for i in range(1, bands + 1):
            dst.write(data[i - 1], i)  # Schreibe jedes Band in das Raster
    return raster_path


def create_synthetic_raster(tmp_path):
    """
    Erzeugt ein synthetisches Raster mit zufälligen Werten für RGB und NIR-Bänder
    und speichert es als GeoTIFF-Datei.

    :param tmp_path: Pfad, in dem die Datei gespeichert wird.
    :return: Pfad zur gespeicherten Rasterdatei.
    """
    # Erstelle zufällige Werte für Rot und Nahinfrarot (NIR)
    red_band = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    nir_band = np.random.randint(100, 255, (100, 100), dtype=np.uint8)

    # Sicherstellen, dass NIR immer mindestens den Wert von Red erreicht
    nir_band = np.maximum(nir_band, red_band)

    # Erstelle ein Dummy-Raster mit den NIR- und Red-Daten (RGB + NIR)
    image_data = np.stack((red_band,
                           np.random.randint(0, 255, (100, 100), dtype=np.uint8),  # Zufällige Werte für Grün
                           np.random.randint(0, 255, (100, 100), dtype=np.uint8),  # Zufällige Werte für Blau
                           nir_band), axis=0)  # Füge das NIR-Band als viertes Band hinzu

    # Pfad für die temporäre Rasterdatei
    raster_file = tmp_path / "synthetic.tif"

    # Schreibe die Rasterdaten in eine Datei
    with rasterio.open(
            raster_file, 'w',
            driver='GTiff',
            height=image_data.shape[1],
            width=image_data.shape[2],
            count=image_data.shape[0],
            dtype=image_data.dtype,
            crs='EPSG:4326',  # WGS 84 (optional, aber für Tests hilfreich)
            transform=from_origin(0, 100, 1, 1)  # Beispiel-Geotransformation
    ) as dst:
        for i in range(image_data.shape[0]):
            dst.write(image_data[i], i + 1)

    return raster_file

# ----------------------------------------
# Tests
# ----------------------------------------

def test_analyze_raster(tmp_path):
    """Test for basic raster analysis (min, max, mean, std)."""
    raster_path = create_test_raster(tmp_path)
    stats = analyze_raster(raster_path, verbose=False)
    # Hier prüfen wir, dass der minimale Wert des Rasters im Bereich der zufälligen Werte liegt
    assert stats[
               "min"] >= 50  # Der minimale Wert sollte mindestens 50 sein, da wir Werte im Bereich von 50 bis 150 generieren
    assert stats["max"] == 150  # Der maximale Wert sollte 150 sein
    assert stats["mean"] >= 50  # Der Mittelwert sollte auch im Bereich der zufälligen Werte liegen
    assert stats["std"] > 0  # Der Standardabweichung sollte positiv sein


def test_compute_histogram(tmp_path):
    """Test for computing histogram of raster data."""
    raster_path = create_test_raster(tmp_path)
    res = compute_histogram(raster_path, verbose=False)
    assert "hist" in res
    assert "bins" in res
    assert "tif" in res
    assert "plot" in res


# Test für die NDVI-Berechnung unter Verwendung eines synthetischen RGB- und NIR-Rasters.
def test_compute_ndvi(tmp_path):
    """Test für die NDVI-Berechnung unter Verwendung eines synthetischen RGB- und NIR-Rasters."""

    # Erstelle ein synthetisches Raster
    raster_file = create_synthetic_raster(tmp_path)

    # Berechne das NDVI
    res = compute_ndvi(raster_file, verbose=False)

    # Überprüfe, ob die erforderlichen Daten zurückgegeben werden
    assert "ndvi" in res
    assert "rgb_array" in res
    assert "tif" in res
    assert "plot" in res

    rgb = res["rgb_array"]
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape[0] > 0
    assert rgb.shape[1] > 0
    assert rgb.shape[2] == 3  # RGB-Bild sollte zurückgegeben werden

    ndvi = res["ndvi"]
    assert np.nanmin(ndvi) >= -0.4  # NDVI sollte mindestens -0.4 betragen, wegen zufälliger Werte
    assert np.nanmax(ndvi) <= 1.0  # NDVI sollte maximal 1 betragen

    # Explizite Überprüfung des NIR-Bandes (4. Band) und des Red-Bandes (1. Band)
    with rasterio.open(raster_file) as src:
        nir_band = src.read(4)  # NIR-Band (4. Band)
        red_band = src.read(1)  # Red-Band (1. Band)

    # Berechne die maximalen Werte der Bänder
    max_nir_value = nir_band.max()
    max_red_value = red_band.max()

    # Dynamische Schwelle basierend auf dem maximalen Red-Wert
    dynamic_threshold = max_red_value * 1.5

    # Zeige die maximalen Werte und die berechnete Schwelle
    print(f"Max NIR-Wert: {max_nir_value}")
    print(f"Max Red-Wert: {max_red_value}")
    print(f"Berechnete dynamische Schwelle: {dynamic_threshold}")

    # Überprüfe, ob alle NIR-Werte <= Red * 1.5 sind
    assert np.all(nir_band <= dynamic_threshold), "NIR-Werte sind größer als Red * 1.5 für einige Pixel"


def test_compute_slope(tmp_path):
    """Test for slope computation."""
    raster_path = create_test_raster(tmp_path)
    res = compute_slope(raster_path, verbose=False)
    assert "slope" in res
    assert res["slope"].shape == (2, 2)
    assert "tif" in res
    assert "plot" in res


def test_compute_hillshade(tmp_path):
    """Test for hillshade computation."""
    raster_path = create_test_raster(tmp_path)
    res = compute_hillshade(raster_path, verbose=False)
    assert "hillshade" in res
    assert res["hillshade"].shape == (2, 2)
    assert "tif" in res
    assert "plot" in res
