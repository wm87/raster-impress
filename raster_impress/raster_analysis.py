import logging
from typing import Optional, Dict, List
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from PIL import Image
from rasterstats import zonal_stats

logger = logging.getLogger("raster_impress")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ----------------------------
# Basic raster statistics
# ----------------------------
def analyze_raster(filepath: str, verbose: bool = True) -> Dict[str, float]:
    with rasterio.open(filepath) as src:
        data = src.read(1)
    stats = {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }
    if verbose:
        logger.info("Raster stats: %s", stats)
    return stats


# ----------------------------
# Histogram
# ----------------------------
def compute_histogram(
    filepath: str,
    output_tif: Optional[str] = None,
    bins: int = 50,
    verbose: bool = True,
) -> Dict[str, object]:
    with rasterio.open(filepath) as src:
        data = src.read(1)
        meta = src.meta.copy()

    hist, bin_edges = np.histogram(data, bins=bins)
    hist_array = np.zeros_like(data, dtype=np.float32)
    for i in range(len(bin_edges) - 1):
        mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
        hist_array[mask] = i

    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_histogram.tif"
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(hist_array, 1)
    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(data, cmap="viridis")
    plt.colorbar(label="Value")
    plt.title("Histogram Preview")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    if verbose:
        logger.info("Histogram TIFF saved to %s, plot saved to %s", output_tif, plot_file)

    return {"hist": hist.tolist(), "bins": bin_edges.tolist(), "tif": output_tif, "plot": plot_file}


# ----------------------------
# NDVI
# ----------------------------
def compute_ndvi(filepath: str, output_tif: Optional[str] = None, verbose: bool = True) -> Dict[str, object]:
    # -------------------------------
    # 1) Raster laden
    # -------------------------------
    with rasterio.open(filepath) as src:
        if src.count < 4:
            raise ValueError("Raster benötigt mindestens 4 Bänder: Rot, Grün, Blau, Infrarot")

        red = src.read(1).astype(np.float32)  # Rotband (Band 1)
        green = src.read(2).astype(np.float32)  # Grünband (Band 2)
        blue = src.read(3).astype(np.float32)  # Blauband (Band 3)
        nir = src.read(4).astype(np.float32)  # Infrarotband (Band 4)

        meta = src.meta.copy()

    # -------------------------------
    # 2) Werte skalieren (0-1)
    # -------------------------------
    for band in [red, green, blue, nir]:
        band /= band.max()  # Normierung auf den Bereich 0-1

    # -------------------------------
    # 3) NDVI berechnen
    # -------------------------------
    denominator = nir + red
    denominator[denominator == 0] = np.nan  # Verhindert Division durch Null
    ndvi = (nir - red) / denominator
    ndvi = np.nan_to_num(ndvi)  # NaN-Werte durch 0 ersetzen

    # -------------------------------
    # 4) Dynamische Schwellenwerte für Landbedeckung bestimmen (Quantile)
    # -------------------------------
    ndvi_forest_thresh = np.quantile(ndvi, 0.75)  # 75% Quantil für Wald
    ndvi_grass_thresh = np.quantile(ndvi, 0.5)  # 50% Quantil für Wiese/Gras
    ndvi_crop_thresh_low = 0.2  # Niedriger Schwellenwert für Acker
    ndvi_crop_thresh_high = 0.4  # Hoher Schwellenwert für Acker

    # -------------------------------
    # 5) Masken erstellen
    # -------------------------------
    forest_mask = (ndvi > ndvi_forest_thresh)
    grass_mask = (ndvi > ndvi_grass_thresh) & (ndvi <= ndvi_forest_thresh)
    crop_mask = (ndvi > ndvi_crop_thresh_low) & (ndvi <= ndvi_crop_thresh_high) & (~forest_mask) & (~grass_mask)
    rock_mask = ~(forest_mask | grass_mask | crop_mask)

    # -------------------------------
    # 6) RGB-Farben zuweisen
    # -------------------------------
    h, w = ndvi.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[forest_mask] = [0, 180, 0]  # Grün für Wald
    rgb[grass_mask] = [255, 255, 0]  # Gelb für Wiese
    rgb[crop_mask] = [210, 180, 140]  # Hellbraun für Acker
    rgb[rock_mask] = [100, 100, 100]  # Dunkelgrau für Felsen/Berge

    # -------------------------------
    # 7) GeoTIFF speichern
    # -------------------------------
    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_landcover_v5.tif"

    meta.update(dtype=rasterio.uint8, count=3)  # 3 Bänder für RGB
    with rasterio.open(output_tif, "w", **meta) as dst:
        for i in range(3):  # 3 Bänder (RGB)
            dst.write(rgb[:, :, i], i + 1)

    # -------------------------------
    # 8) PNG speichern
    # -------------------------------
    png_path = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    if verbose:
        print("TIF saved:", output_tif)
        print("PNG saved:", png_path)

    # -------------------------------
    # 9) Rückgabe
    # -------------------------------
    return {
        "ndvi": ndvi,
        "tif": output_tif,
        "plot": png_path,
        "rgb_array": rgb
    }

# ----------------------------
# 1) Slope berechnen
# ----------------------------
def compute_slope(filepath: str, output_tif: str = None, verbose: bool = True) -> dict:

    with rasterio.open(filepath) as src:
        height = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform

    height[height <= -9999] = np.nan
    res_x = transform.a
    res_y = -transform.e

    # Slope berechnen
    dy, dx = np.gradient(height, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_norm = (slope_deg - np.nanmin(slope_deg)) / (np.nanmax(slope_deg) - np.nanmin(slope_deg))

    # Dateinamen festlegen, falls None
    prefix = os.path.splitext(filepath)[0]
    png_file = prefix + "_slope.png"
    if output_tif is None:
        output_tif = prefix + "_slope.tif"

    # PNG speichern
    Image.fromarray((slope_norm * 255).astype(np.uint8)).save(png_file)

    # GeoTIFF speichern
    profile.update(dtype="float32", count=1)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(slope_deg.astype(np.float32), 1)

    if verbose:
        print(f"Slope gespeichert: {png_file}, {output_tif}")

    return {"slope_deg": slope_deg, "png": png_file, "tif": output_tif}

# ----------------------------
# 2) Hillshade berechnen
# ----------------------------
def compute_hillshade(filepath: str, output_tif: str = None, azimuth: float = 315, altitude: float = 45, verbose: bool = True) -> dict:

    with rasterio.open(filepath) as src:
        height = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform

    height[height <= -9999] = np.nan
    res_x = transform.a
    res_y = -transform.e

    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    hillshade = ls.hillshade(height, vert_exag=2.0, dx=res_x, dy=res_y)

    prefix = os.path.splitext(filepath)[0]
    png_file = prefix + "_hillshade.png"
    if output_tif is None:
        output_tif = prefix + "_hillshade.tif"

    Image.fromarray((hillshade * 255).astype(np.uint8)).save(png_file)

    profile.update(dtype="uint8", count=1)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write((hillshade * 255).astype(np.uint8), 1)

    if verbose:
        print(f"Hillshade gespeichert: {png_file}, {output_tif}")

    return {"hillshade": hillshade, "png": png_file, "tif": output_tif}

# ----------------------------
# 3) Plastisches Relief berechnen
# ----------------------------
def compute_relief(filepath: str, output_tif: str = None, verbose: bool = True) -> dict:

    # -------------------------
    # DEM laden
    # -------------------------
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(float)
        profile = src.profile.copy()
        transform = src.transform

    dem = np.where(dem <= -9999, np.nan, dem)
    res_x = transform.a
    res_y = -transform.e

    # -------------------------
    # Hillshade berechnen
    # -------------------------
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=2.0, dx=res_x, dy=res_y)

    # -------------------------
    # Relieffarbe (terrain)
    # -------------------------
    h_min, h_max = np.nanmin(dem), np.nanmax(dem)
    dem_norm = (dem - h_min) / (h_max - h_min)
    cmap = plt.cm.terrain
    relief_color = cmap(dem_norm)[:, :, :3]

    # Plastisches Relief = Farbe × Hillshade
    relief = relief_color * hillshade[..., np.newaxis]
    relief_uint8 = (relief * 255).astype(np.uint8)

    # -------------------------
    # Dateipfade
    # -------------------------
    prefix = os.path.splitext(filepath)[0]
    png_file = prefix + "_relief_plastisch.png"
    if output_tif is None:
        output_tif = prefix + "_relief_plastisch.tif"

    # -------------------------
    # PNG speichern
    # -------------------------
    Image.fromarray(relief_uint8).save(png_file)

    # -------------------------
    # GeoTIFF speichern (RGB)
    # -------------------------
    profile.update(dtype="uint8", count=3)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(relief_uint8[:, :, 0], 1)
        dst.write(relief_uint8[:, :, 1], 2)
        dst.write(relief_uint8[:, :, 2], 3)

    if verbose:
        print(f"Relief gespeichert: {png_file}, {output_tif}")

    return {"relief_png": png_file, "relief_tif": output_tif, "relief_array": relief}

# ----------------------------
# Metadata
# ----------------------------
def print_metadata(filepath: str, verbose: bool = True) -> Dict:
    with rasterio.open(filepath) as src:
        metadata = {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "transform": str(src.transform),
            "nodata": src.nodatavals,
        }
        for i in range(1, src.count + 1):
            band = src.read(i)
            metadata[f"band_{i}"] = {
                "min": float(band.min()),
                "max": float(band.max()),
                "mean": float(band.mean()),
            }
    if verbose:
        logger.info("Metadata: %s", metadata)
    return metadata

# ----------------------------
# Quality Check
# ----------------------------
def quality_check(filepath: str, verbose: bool = True) -> List[Dict]:
    results = []
    with rasterio.open(filepath) as src:
        data_shape = None
        for i in range(1, src.count + 1):
            band = src.read(i)
            result = {"band": i, "nodata_count": 0, "shape_ok": True}

            if data_shape is None:
                data_shape = band.shape
            elif band.shape != data_shape:
                result["shape_ok"] = False
                if verbose:
                    logger.warning("Band %d shape %s != first band %s", i, band.shape, data_shape)

            nodata = src.nodatavals[i - 1]
            if nodata is not None:
                nodata_count = int(np.sum(band == nodata))
                result["nodata_count"] = nodata_count
                result["nodata_percent"] = 100 * nodata_count / band.size
                if verbose:
                    logger.info(
                        "Band %d NoData pixels: %d (%.2f%%)", i, nodata_count, result["nodata_percent"]
                    )
            results.append(result)
    return results

# ----------------------------
# Zonal Stats
# ----------------------------
def compute_zonal_stats(raster_path: str, vector_path: str,
                        stats_list: Optional[List[str]] = None,
                        verbose: bool = True) -> List[Dict]:
    if stats_list is None:
        stats_list = ["mean", "min", "max"]
    zs = zonal_stats(vector_path, raster_path, stats=stats_list, geojson_out=True)
    if verbose:
        for i, zone in enumerate(zs):
            logger.info("Zone %d: %s", i, zone["properties"])
    return zs
