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


def create_dummy_raster(path, count=4, dtype="uint8", width=10, height=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(np.dtype(dtype), np.integer):
        data = np.random.randint(0, 255, size=(count, height, width), dtype=dtype)
    else:
        data = np.random.random((count, height, width)).astype(dtype)

    # Realistischer Transform für EPSG:25833
    transform = from_origin(500000, 5800000, 1, 1)

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


def test_cli_extract_option(data_dir: Path, tmp_path: Path):
    """Testet die Feature-Extraction (--extract) mit DOP + DSM + DGM."""

    # Dummy-Raster erzeugen
    dop = create_dummy_raster(tmp_path / "dop.tif", count=4, dtype="uint8")      # 4 Bänder: R,G,B,NIR
    dsm = create_dummy_raster(tmp_path / "dsm.tif", count=1, dtype="float32")    # 1 Band
    dgm = create_dummy_raster(tmp_path / "dgm.tif", count=1, dtype="float32")    # 1 Band

    # Sicherstellen, dass Dateien existieren
    assert dop.exists(), "DOP existiert nicht"
    assert dsm.exists(), "DSM existiert nicht"
    assert dgm.exists(), "DGM existiert nicht"

    # Ausgabeordner
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)

    argv = [
        str(dop),
        "--extract", str(dsm), str(dgm),
        "--output", str(output_dir)
    ]

    # CLI aufrufen
    results = main(argv)

    # Ergebnisse prüfen
    assert "extract" in results, "Key 'extract' fehlt in den Ergebnissen"
    assert isinstance(results["extract"], dict), "extract ist kein Dictionary"

    # Dateien existieren prüfen
    for key, path_str in results["extract"].items():
        path = Path(path_str)
        assert path.exists(), f"{key} Shapefile wurde nicht erzeugt: {path}"
