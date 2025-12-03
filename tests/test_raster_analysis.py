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


def create_raster(tmp_path, bands=1):
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "test.tif"
    with rasterio.open(
        path,
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
    return path


def test_analyze_raster(tmp_path):
    raster_path = create_raster(tmp_path)
    stats = analyze_raster(raster_path, verbose=False)
    assert stats["min"] == 1
    assert stats["max"] == 4
    assert stats["mean"] == 2.5
    assert stats["std"] > 0


def test_compute_histogram(tmp_path):
    raster_path = create_raster(tmp_path)
    res = compute_histogram(raster_path, verbose=False)
    # In der aktuellen Implementation sind die Keys: hist, bins, tif, plot
    assert "hist" in res
    assert "bins" in res
    assert "tif" in res
    assert "plot" in res


def test_compute_ndvi(tmp_path):
    raster_path = create_raster(tmp_path, bands=2)
    res = compute_ndvi(raster_path, verbose=False)
    assert "ndvi" in res
    assert res["ndvi"].shape == (2, 2)
    assert "tif" in res
    assert "plot" in res


def test_compute_slope(tmp_path):
    raster_path = create_raster(tmp_path)
    res = compute_slope(raster_path, verbose=False)
    assert "slope" in res
    assert res["slope"].shape == (2, 2)
    assert "tif" in res
    assert "plot" in res


def test_compute_hillshade(tmp_path):
    raster_path = create_raster(tmp_path)
    res = compute_hillshade(raster_path, verbose=False)
    assert "hillshade" in res
    assert res["hillshade"].shape == (2, 2)
    assert "tif" in res
    assert "plot" in res


def test_print_metadata(tmp_path):
    raster_path = create_raster(tmp_path)
    metadata = print_metadata(raster_path, verbose=False)
    assert metadata["width"] == 2
    assert metadata["height"] == 2
    assert metadata["bands"] == 1
    assert "band_1" in metadata


def test_quality_check(tmp_path):
    raster_path = create_raster(tmp_path)
    results = quality_check(raster_path, verbose=False)
    assert len(results) == 1
    assert results[0]["band"] == 1
    assert results[0]["shape_ok"] is True
    assert results[0]["nodata_count"] == 0
