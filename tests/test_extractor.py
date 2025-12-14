from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from raster_impress.extractor import (
    create_impervious_mask,
    save_shapefile,
    save_impervious_surfaces_shapefile,
    extract_all_features,
)
from test_helpers import create_dummy_raster, create_dummy_dem

# -----------------------------
# Testklasse
# -----------------------------
class TestExtractor:

    @pytest.fixture
    def temp_rasters(self, tmp_path):
        """Erstellt Dummy-Rasterdateien für DOP, DSM, DGM und Output-Ordner."""
        dop_file = create_dummy_raster(tmp_path / "dop.tif", count=4)
        dsm_file = create_dummy_dem(tmp_path / "dsm.tif", value=2.0)
        dgm_file = create_dummy_dem(tmp_path / "dgm.tif", value=1.5)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        return dop_file, dsm_file, dgm_file, out_dir

    # -----------------------------
    # Test: create_impervious_mask
    # -----------------------------
    def test_create_impervious_mask(self):
        dsm = np.array([[2, 3], [4, 5]], dtype=np.float32)
        dgm = np.array([[1, 2], [3, 4]], dtype=np.float32)
        transform = from_origin(0, 2, 1, 1)
        mask = create_impervious_mask(dsm, dgm, transform)
        assert mask.shape == dsm.shape
        assert mask.dtype == np.uint8
        assert np.all((mask == 0) | (mask == 1))

    # -----------------------------
    # Test: save_shapefile und save_impervious_surfaces_shapefile
    # -----------------------------
    def test_save_shapefiles(self, tmp_path):
        mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        transform = from_origin(0, 2, 1, 1)
        crs = "EPSG:4326"

        shp_file = tmp_path / "test.shp"
        save_shapefile(mask, transform, crs, shp_file)
        assert shp_file.exists()
        for ext in ["shx", "dbf", "prj"]:
            assert (tmp_path / f"test.{ext}").exists()

        # Test: filtered impervious surfaces
        shp_file2 = tmp_path / "impervious.shp"
        save_impervious_surfaces_shapefile(mask, transform, crs, shp_file2, min_area_m2=0.5)
        assert shp_file2.exists()
        for ext in ["shx", "dbf", "prj"]:
            assert (tmp_path / f"impervious.{ext}").exists()

    # -----------------------------
    # Test: extract_all_features
    # -----------------------------
    def test_extract_all_features(self, temp_rasters):
        dop_file, dsm_file, dgm_file, out_dir = temp_rasters

        results = extract_all_features(
            dop_raster=str(dop_file),
            dsm_raster=str(dsm_file),
            dgm_raster=str(dgm_file),
            output_base=str(out_dir),
            red_channel_raster=None,
            verbose=False
        )

        # Prüfen, dass alle Keys existieren
        for key in ["vegetation", "impervious_surfaces", "buildings"]:
            assert key in results
            path = Path(results[key])
            assert path.exists()
            for ext in ["shx", "dbf", "prj"]:
                assert (path.parent / f"{path.stem}.{ext}").exists()
