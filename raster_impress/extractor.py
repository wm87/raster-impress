import glob
import logging
import os

import fiona
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import binary_closing
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

# Logger
logger = logging.getLogger("raster_impress")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ======================================================
# 0️⃣ Output-Ordner säubern
# ======================================================
def cleanup_output(output_base, verbose: bool = True):
    os.makedirs(output_base, exist_ok=True)
    for ext in ["shp", "shx", "dbf", "prj", "cpg", "tif", "png"]:
        for f in glob.glob(os.path.join(output_base, f"*{ext}")):
            os.remove(f)
    if verbose:
        logger.info(f"✔ Output-Ordner gelöscht: {output_base}")

# ======================================================
# 1️⃣ NDVI berechnen
# ======================================================
def compute_ndvi(filepath):
    with rasterio.open(filepath) as src:
        r = src.read(1).astype(np.float32)
        nir = src.read(4).astype(np.float32)
        transform = src.transform
        crs = src.crs
        profile = src.profile

    ndvi = (nir - r) / (nir + r)
    ndvi[(nir + r) == 0] = np.nan
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi, transform, crs, profile

# ======================================================
# 2️⃣ Raster reprojezieren + resamplen auf NDVI-Grid (Pixelgenau!)
# ======================================================
def reproject_to_match(path, ref_transform, ref_crs, ref_shape):
    with rasterio.open(path) as src:
        dst = np.zeros(ref_shape, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )

    dst[dst == src.nodata] = np.nan
    return dst

# ======================================================
# 3️⃣ Impervious surfaces extrahieren + kleine Löcher schließen
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
# 4️⃣ Shapefile speichern
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
# 4b️⃣ Impervious surfaces-Shapefile speichern mit Flächenfilter
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
# 5️⃣ Hauptprozess
# ======================================================
def extract_all_features(dop_raster, dsm_raster, dgm_raster, output_base, red_channel_raster=None, verbose: bool = True):
    cleanup_output(output_base, verbose=verbose)

    # --- NDVI ---
    ndvi, ref_transform, ref_crs, ref_profile = compute_ndvi(dop_raster)
    shape_y, shape_x = ndvi.shape
    ref_shape = (shape_y, shape_x)

    # Vegetation extrahieren
    vegetation_mask = (ndvi > 0.3).astype(np.uint8)
    save_shapefile(vegetation_mask, ref_transform, ref_crs,
                   os.path.join(output_base, "vegetation.shp"), verbose=verbose)

    # --- DSM/DGM auf NDVI reprojezieren ---
    dsm = reproject_to_match(dsm_raster, ref_transform, ref_crs, ref_shape)
    dgm = reproject_to_match(dgm_raster, ref_transform, ref_crs, ref_shape)

    red_channel = None
    if red_channel_raster:
        red_channel = reproject_to_match(red_channel_raster, ref_transform, ref_crs, ref_shape)

    # --- Impervious surfaces ---
    impervious_mask = create_impervious_mask(dsm, dgm, ref_transform, red_channel, min_hole_area_m2=20)
    save_impervious_surfaces_shapefile(impervious_mask, ref_transform, ref_crs,
                                       os.path.join(output_base, "impervious_surfaces.shp"),
                                       min_area_m2=20, verbose=verbose)

    # --- Buildings = invertierte Vereinigung ---
    combined = np.logical_or(vegetation_mask == 1, impervious_mask == 1)
    building_mask = (~combined).astype(np.uint8)
    building_mask[np.isnan(ndvi)] = 0  # keine weißen Pixel außerhalb des Bildes

    save_shapefile(building_mask, ref_transform, ref_crs,
                   os.path.join(output_base, "buildings.shp"), verbose=verbose)

    if verbose:
        logger.info("✔ Fertig: vegetation, impervious surfaces, buildings sind pixelgenau ausgerichtet.")
