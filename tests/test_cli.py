from pathlib import Path
import pytest
from raster_impress.cli import main
from tests.test_helpers import create_dummy_raster

@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Fixture für temporäres Testdatenverzeichnis."""
    return tmp_path

def test_cli_all_options(data_dir: Path):
    # Dummy-Raster erzeugen (4 Bänder für NDVI)
    raster_file = create_dummy_raster(data_dir / "dummy_full.tif", count=4, dtype="uint8")
    output_dir = data_dir

    argv = [
        str(raster_file),
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

    results = main(argv, output_dir=output_dir)

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

    # Prüfen, dass alle Ausgabedateien existieren
    for suffix in ["hist", "ndvi", "slope", "hillshade", "relief_plastisch"]:
        tif_file = output_dir / f"{raster_file.stem}_{suffix}.tif"
        png_file = output_dir / f"{raster_file.stem}_{suffix}.png"
        assert tif_file.exists()
        assert png_file.exists()
