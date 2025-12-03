import os
import logging
from typing import Optional, Dict, List
import numpy as np
import rasterio
import matplotlib.pyplot as plt
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
    with rasterio.open(filepath) as src:
        if src.count < 2:
            raise ValueError("NDVI requires at least 2 bands (Red, NIR)")
        red = src.read(1).astype(np.float32)
        nir = src.read(2).astype(np.float32)
        meta = src.meta.copy()

    ndvi = (nir - red) / (nir + red + 1e-10)
    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_ndvi.tif"
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(ndvi, 1)
    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(ndvi, cmap="RdYlGn")
    plt.colorbar(label="NDVI")
    plt.title("NDVI")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    if verbose:
        logger.info("NDVI saved to %s, plot saved to %s", output_tif, plot_file)

    return {"ndvi": ndvi, "tif": output_tif, "plot": plot_file}


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
                      azimuth: float = 315, altitude: float = 45,
                      verbose: bool = True) -> Dict[str, object]:
    with rasterio.open(filepath) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        meta = src.meta.copy()
        xres = transform.a
        yres = -transform.e

    dzdx = np.gradient(dem, axis=1) / xres
    dzdy = np.gradient(dem, axis=0) / yres
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    hillshade = np.sin(alt_rad) * np.sin(slope_rad) + \
                np.cos(alt_rad) * np.cos(slope_rad) * np.cos(az_rad - aspect_rad)
    hillshade = np.clip(hillshade, 0, 1)

    if output_tif is None:
        output_tif = os.path.splitext(filepath)[0] + "_hillshade.tif"
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(hillshade, 1)

    plot_file = os.path.splitext(output_tif)[0] + ".png"
    plt.figure(figsize=(12, 8))
    plt.imshow(hillshade, cmap="gray")
    plt.title("Hillshade")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    if verbose:
        logger.info("Hillshade saved to %s, plot saved to %s", output_tif, plot_file)

    return {"hillshade": hillshade, "tif": output_tif, "plot": plot_file}


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
