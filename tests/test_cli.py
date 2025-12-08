from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from raster_impress.cli import main


@pytest.fixture
def data_dir() -> Path:
    """Verzeichnis für dauerhaft gespeicherte Testdaten."""
    path = Path(__file__).parent / "test_data"
    path.mkdir(exist_ok=True)
    return path


def create_dummy_raster(path: Path, count=1, dtype="uint8", width=10, height=10) -> Path:
    """Erstellt ein Dummy-Raster, das mit Rasterio gelesen werden kann."""

    # Verzeichnis sicherstellen
    path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy-Daten erzeugen
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = np.random.randint(0, 255, size=(count, height, width), dtype=dtype)
    elif np.issubdtype(np.dtype(dtype), np.floating):
        data = np.random.random(size=(count, height, width)).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Transform definieren (links oben Ecke bei 0,0, Pixelgröße=1)
    transform = from_origin(0, 0, 1, 1)

    # GeoTIFF schreiben
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:25833",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


def test_cli_options(data_dir: Path):
    """Testet alle Analyseoptionen der CLI."""
    raster = create_dummy_raster(data_dir / "dop.tif", count=4, dtype="uint8")
    output_dir = data_dir

    argv = [
        str(raster),
        "--output", str(output_dir),
        "--stats",
        "--histogram",
        "--ndvi",
        "--slope",
        "--hillshade",
        "--relief",
        "--metadata",
        "--quality",
        "--silent",
    ]

    results = main(argv)

    expected_keys = [
        "stats",
        "histogram",
        "ndvi",
        "slope",
        "hillshade",
        "relief",
        "metadata",
        "quality",
    ]

    for key in expected_keys:
        assert key in results

    # Prüfen, dass alle Ausgabedateien existieren (.tif)
    for suffix in ["hist", "ndvi", "slope", "hillshade", "relief_plastisch"]:
        tif = output_dir / f"{raster.stem}_{suffix}.tif"
        assert tif.exists()


def test_cli_extract_option(data_dir: Path):
    """Testet die Feature-Extraction (--extract) mit DOP + DSM + DGM."""

    # DOP-Raster: 4 Bänder (R,G,B,NIR)
    dop = create_dummy_raster(data_dir / "dop.tif", count=4, dtype="uint8")

    # DSM / DGM: 1 Band, float32
    dsm = create_dummy_raster(data_dir / "dsm.tif", count=1, dtype="float32")
    dgm = create_dummy_raster(data_dir / "dgm.tif", count=1, dtype="float32")

    output_dir = data_dir

    # Dateien existieren prüfen
    assert dop.exists(), "DOP existiert nicht"
    assert dsm.exists(), "DSM existiert nicht"
    assert dgm.exists(), "DGM existiert nicht"

    argv = [
        str(dop),
        "--extract", str(dsm), str(dgm),
        "--output", str(output_dir),
        "--silent"
    ]

    results = main(argv)

    assert "extract" in results
    assert isinstance(results["extract"], dict)
