import logging
from pathlib import Path

import fiona
import numpy as np
from rasterio import features
from scipy.ndimage import binary_closing
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

from raster_impress.helper import cleanup_output, reproject_to_match
from raster_impress.raster_analysis import compute_ndvi

# Logger
logger = logging.getLogger("raster_impress")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ======================================================
# Impervious Surfaces extrahieren + kleine Löcher schließen
# ======================================================
def create_impervious_mask(dsm, dgm, transform, red_channel=None, min_hole_area_m2=20):
    height_diff = dsm - dgm
    flat_mask = (height_diff <= 1).astype(np.uint8)

    if red_channel is not None:
        mask = flat_mask & (red_channel < 100)
    else:
        mask = flat_mask

    # Pixelgröße
    px = abs(transform.a)
    py = abs(transform.e)

    # Anzahl Pixel für min_hole_area
    pixels_min_area = int(np.ceil(np.sqrt(min_hole_area_m2 / (px * py))))

    # Löcher < min_hole_area_m2 schließen
    structure = np.ones((pixels_min_area, pixels_min_area))
    mask = binary_closing(mask, structure=structure).astype(np.uint8)

    return mask

# ======================================================
# Shape speichern
# ======================================================
def save_shapefile(mask, transform, crs, path, verbose: bool = True):
    polygons = [shape(geom) for geom, value in features.shapes(mask, mask == 1, transform=transform) if value == 1]

    if not polygons:
        if verbose:
            logger.warning(f"⚠ Keine Polygone für {path}")
        return

    merged = unary_union(polygons)
    schema = {"geometry": "Polygon", "properties": {"id": "int"}}

    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as shp:
        if merged.geom_type == "Polygon":
            shp.write({"geometry": mapping(merged), "properties": {"id": 1}})
        else:
            for i, poly in enumerate(merged.geoms, 1):
                shp.write({"geometry": mapping(poly), "properties": {"id": i}})

    if verbose:
        logger.info(f"✔ Shapefile gespeichert: {path}")

# ======================================================
# Impervious Surfaces speichern (inkl. Flächenfilter)
# ======================================================
def save_impervious_surfaces_shapefile(mask, transform, crs, path, min_area_m2=20, verbose: bool = True):
    polygons = [shape(geom) for geom, value in features.shapes(mask, mask == 1, transform=transform) if value == 1]

    if not polygons:
        if verbose:
            logger.warning(f"⚠ Keine Impervious surfaces-Polygone für {path}")
        return

    merged = unary_union(polygons)
    poly_list = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)

    filtered_polys = [poly for poly in poly_list if poly.area >= min_area_m2]

    if not filtered_polys:
        if verbose:
            logger.warning(f"⚠ Keine Impervious surfaces-Polygone >= {min_area_m2} m² für {path}")
        return

    schema = {"geometry": "Polygon", "properties": {"id": "int"}}
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as shp:
        for i, poly in enumerate(filtered_polys, 1):
            shp.write({"geometry": mapping(poly), "properties": {"id": i}})

    if verbose:
        logger.info(f"✔ Impervious surfaces-Shapefile gespeichert: {path}, {len(filtered_polys)} Polygone")

# ======================================================
# Hauptprozess
# ======================================================
def extract_all_features(dop_raster, dsm_raster, dgm_raster, output_base,
                         red_channel_raster=None, verbose: bool = True):
    dop_raster = Path(dop_raster).resolve()
    dsm_raster = Path(dsm_raster).resolve()
    dgm_raster = Path(dgm_raster).resolve()
    output_base = Path(output_base).resolve()

    if not dop_raster.exists():
        raise FileNotFoundError(f"dop_raster not found: {dop_raster}")
    if not dsm_raster.exists():
        raise FileNotFoundError(f"dsm_raster not found: {dsm_raster}")
    if not dgm_raster.exists():
        raise FileNotFoundError(f"dgm_raster not found: {dgm_raster}")

    output_base.mkdir(parents=True, exist_ok=True)
    cleanup_output(output_base, verbose=verbose)

    # --- NDVI ---
    try:
        result = compute_ndvi(str(dop_raster))
        ndvi = result["ndvi"]
        ref_transform = result["transform"]
        ref_crs = result["crs"]
    except Exception as e:
        raise RuntimeError(f"Fehler beim Berechnen von NDVI: {e}")

    # Jetzt kannst du die Form von ndvi korrekt verwenden
    shape_y, shape_x = ndvi.shape
    ref_shape = (shape_y, shape_x)

    vegetation_mask = np.array(ndvi >= 0.2, dtype=np.uint8)
    save_shapefile(vegetation_mask, ref_transform, ref_crs, output_base / "vegetation.shp", verbose=verbose)

    dsm = reproject_to_match(dsm_raster, ref_transform, ref_crs, ref_shape)
    dgm = reproject_to_match(dgm_raster, ref_transform, ref_crs, ref_shape)

    red_channel = None
    if red_channel_raster:
        red_channel = reproject_to_match(red_channel_raster, ref_transform, ref_crs, ref_shape)

    impervious_mask = create_impervious_mask(dsm, dgm, ref_transform, red_channel, min_hole_area_m2=20)
    save_impervious_surfaces_shapefile(impervious_mask, ref_transform, ref_crs,
                                       output_base / "impervious_surfaces.shp",
                                       min_area_m2=20, verbose=verbose)

    combined = np.logical_or(vegetation_mask == 1, impervious_mask == 1)
    building_mask = (~combined).astype(np.uint8)
    building_mask[np.isnan(ndvi)] = 0

    save_shapefile(building_mask, ref_transform, ref_crs,
                   output_base / "buildings.shp", verbose=verbose)

    return {"vegetation": str(output_base / "vegetation.shp"),
            "impervious_surfaces": str(output_base / "impervious_surfaces.shp"),
            "buildings": str(output_base / "buildings.shp")
    }
