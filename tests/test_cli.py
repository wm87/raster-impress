import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from raster_impress import cli


def create_test_raster(tmp_path: Path, bands=1):
    """Helper: create small raster with given number of bands"""
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    raster_path = tmp_path / "test.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=bands,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        for i in range(1, bands + 1):
            dst.write(data, i)
    return raster_path


def run_cli_args(tmp_path, *args):
    """Helper: set sys.argv and call cli.main()"""
    raster_path = create_test_raster(tmp_path, bands=2 if "--ndvi" in args else 1)
    sys_argv_backup = sys.argv
    sys.argv = ["cli", str(raster_path), *args]
    try:
        result = cli.main()
    finally:
        sys.argv = sys_argv_backup
    return result


def test_cli_stats(tmp_path):
    result = run_cli_args(tmp_path, "--stats")
    assert "stats" in result
    stats = result["stats"]
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats


def test_cli_histogram(tmp_path):
    result = run_cli_args(tmp_path, "--histogram")
    assert "histogram" in result
    hist = result["histogram"]
    assert "tif" in hist or "plot" in hist


def test_cli_ndvi(tmp_path):
    result = run_cli_args(tmp_path, "--ndvi")
    assert "ndvi" in result
    ndvi = result["ndvi"]
    assert "tif" in ndvi or "plot" in ndvi


def test_cli_slope(tmp_path):
    result = run_cli_args(tmp_path, "--slope")
    assert "slope" in result
    slope = result["slope"]
    assert "tif" in slope or "plot" in slope


def test_cli_hillshade(tmp_path):
    result = run_cli_args(tmp_path, "--hillshade")
    assert "hillshade" in result
    hs = result["hillshade"]
    assert "tif" in hs or "plot" in hs


def test_cli_metadata(tmp_path):
    result = run_cli_args(tmp_path, "--metadata")
    assert "metadata" in result
    meta = result["metadata"]
    assert "width" in meta
    assert "height" in meta
    assert "bands" in meta


def test_cli_quality(tmp_path):
    result = run_cli_args(tmp_path, "--quality")
    assert "quality" in result
    quality = result["quality"]
    assert isinstance(quality, list)
