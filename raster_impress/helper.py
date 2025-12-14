import glob
import logging
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

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
