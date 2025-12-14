from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_raster(
    filepath: Path | str,
    width: int = 10,
    height: int = 10,
    count: int = 1,
    dtype: str = "uint8",
    nodata: int = 0
) -> Path:
    """Erzeugt Dummy-Rasterdatei für Tests."""
    filepath = Path(filepath)
    transform = from_origin(0, 0, 1, 1)

    if np.issubdtype(np.dtype(dtype), np.integer):
        data = np.random.randint(0, 255, size=(count, height, width), dtype=dtype)
    else:
        data = np.random.random(size=(count, height, width)).astype(dtype)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="+proj=latlong",
        transform=transform,
        nodata=nodata
    ) as dst:
        for i in range(count):
            dst.write(data[i], i + 1)

    return filepath


def create_dummy_dem(path, shape=(10, 10), value=2.0):
    from rasterio.transform import from_origin
    import rasterio
    data = np.full(shape, value, dtype=np.float32)
    transform = from_origin(0, shape[0], 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=shape[0], width=shape[1],
        count=1, dtype=np.float32,
        crs="EPSG:4326",
        transform=transform
    ) as dst:
        dst.write(data, 1)
    return path
