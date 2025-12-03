from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("raster_impress")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback
