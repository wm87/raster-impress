from pathlib import Path
from raster_impress.raster_analysis import (
    analyze_raster,
    compute_histogram,
    compute_ndvi,
    compute_slope,
    compute_hillshade,
    compute_relief,
)
from test_helpers import create_dummy_raster


def test_analyze_raster(data_dir: Path):
    raster_file = create_dummy_raster(data_dir / "dummy_stats.tif", count=1)
    result = analyze_raster(str(raster_file))
    assert isinstance(result, dict)

def test_compute_histogram(data_dir: Path):
    raster_file = create_dummy_raster(data_dir / "dummy_hist.tif", count=1)
    output_tif = data_dir / "dummy_hist_output.tif"
    result = compute_histogram(str(raster_file), output_tif=str(output_tif))
    assert output_tif.exists()
    assert "hist" in result

def test_compute_ndvi(data_dir: Path):
    raster_file = create_dummy_raster(data_dir / "dummy_ndvi.tif", count=4)
    output_tif = data_dir / "dummy_ndvi_output.tif"
    result = compute_ndvi(str(raster_file), output_tif=str(output_tif))
    assert output_tif.exists()
    assert "ndvi" in result

def test_compute_slope(data_dir: Path):
    dem_file = create_dummy_raster(data_dir / "dummy_dem.tif", count=1, dtype="float32")
    output_tif = data_dir / "dummy_slope.tif"
    result = compute_slope(str(dem_file), output_tif=str(output_tif))
    assert output_tif.exists()
    assert "slope_deg" in result

def test_compute_hillshade(data_dir: Path):
    dem_file = create_dummy_raster(data_dir / "dummy_dem_hs.tif", count=1, dtype="float32")
    output_tif = data_dir / "dummy_hillshade.tif"
    result = compute_hillshade(str(dem_file), output_tif=str(output_tif))
    assert output_tif.exists()
    assert "hillshade" in result

def test_compute_relief(data_dir: Path):
    dem_file = create_dummy_raster(data_dir / "dummy_dem_relief.tif", count=1, dtype="float32")
    output_tif = data_dir / "dummy_relief.tif"
    result = compute_relief(str(dem_file), output_tif=str(output_tif))
    assert output_tif.exists()
    assert "relief_array" in result