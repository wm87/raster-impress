import logging
import os
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rasterio
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
# Slope
# ----------------------------
def compute_slope(filepath: str, output_tif: Optional[str] = None, verbose: bool = True) -> Dict[str, object]:
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        meta = src.meta.copy()
        xres = transform.a
        yres = -transform.e

    dzdx = np.gradient(dem, axis=1) / xres
    dzdy = np.gradient(dem, axis=0) / yres
    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2)) * 180 / np.pi

    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_slope.tif"
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(slope, 1)

    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(slope, cmap="terrain")
    plt.colorbar(label="Slope [deg]")
    plt.title("Slope")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    if verbose:
        logger.info("Slope saved to %s, plot saved to %s", output_tif, plot_file)

    return {"slope": slope, "tif": output_tif, "plot": plot_file}


# ----------------------------
# Hillshade
# ----------------------------
def compute_hillshade(filepath: str, output_tif: Optional[str] = None,
                      azimuth: float = 315, altitude: float = 30,
                      gamma: float = 0.9,
                      enhance_slope: bool = True,
                      verbose: bool = True) -> Dict[str, object]:

    # DEM einlesen
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        meta = src.meta.copy()
        xres = transform.a
        yres = -transform.e

    # Gradienten berechnen
    dzdx = np.gradient(dem, axis=1) / xres
    dzdy = np.gradient(dem, axis=0) / yres
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdy, -dzdx)

    # Hillshade berechnen
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hillshade = np.sin(alt_rad) * np.sin(slope_rad) + \
                np.cos(alt_rad) * np.cos(slope_rad) * np.cos(az_rad - aspect_rad)

    # Optional: Hangneigung zur Verstärkung hinzufügen
    if enhance_slope:
        hillshade = hillshade * 0.6 + (slope_rad / np.pi) * 0.4

    # Normalisieren auf 0–1
    hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())

    # Gamma-Korrektur
    hillshade = np.power(hillshade, gamma)

    # Auf 0–255 skalieren für TIFF
    hillshade_uint8 = (hillshade * 255).astype(np.uint8)

    # TIFF speichern
    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_hillshade.tif"
    meta.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(hillshade_uint8, 1)

    # PNG-Plot speichern (identisch zum TIFF)
    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(hillshade_uint8, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if verbose:
        logger.info("Hillshade saved to %s, plot saved to %s", output_tif, plot_file)

    return {"hillshade": hillshade_uint8, "tif": output_tif, "plot": plot_file}

def compute_hillshade_auto(filepath: str, output_tif: Optional[str] = None,
                           azimuth: float = 315, altitude: float = 30,
                           verbose: bool = True) -> Dict[str, object]:

    # DEM einlesen
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        meta = src.meta.copy()
        xres = transform.a
        yres = -transform.e

    # Gradienten berechnen
    dzdx = np.gradient(dem, axis=1) / xres
    dzdy = np.gradient(dem, axis=0) / yres
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdy, -dzdx)

    # Hillshade berechnen
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hillshade = np.sin(alt_rad) * np.sin(slope_rad) + \
                np.cos(alt_rad) * np.cos(slope_rad) * np.cos(az_rad - aspect_rad)

    # Automatische Verstärkung: Hangneigung einbeziehen
    hillshade = hillshade * 0.6 + (slope_rad / np.pi) * 0.4

    # Werte auf 0–1 normalisieren
    hillshade = (hillshade - np.min(hillshade)) / (np.max(hillshade) - np.min(hillshade))

    # Automatischer Gamma-Boost basierend auf Histogramm
    # Idee: je kleiner der Kontrast, desto stärker Gamma < 1
    contrast = np.percentile(hillshade, 95) - np.percentile(hillshade, 5)
    gamma = np.clip(0.5 + 0.5 * contrast, 0.6, 0.9)  # kleiner Kontrast -> Gamma kleiner
    hillshade = np.power(hillshade, gamma)

    # Auf 0–255 skalieren
    hillshade_uint8 = (hillshade * 255).astype(np.uint8)

    # TIFF speichern
    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_hillshade.tif"
    meta.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(hillshade_uint8, 1)

    # PNG-Plot speichern (identisch zum TIFF)
    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(hillshade_uint8, cmap="gray")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if verbose:
        logger.info("Auto Hillshade saved to %s, plot saved to %s (gamma=%.2f)", output_tif, plot_file, gamma)

    return {"hillshade": hillshade_uint8, "tif": output_tif, "plot": plot_file, "gamma": gamma}
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
