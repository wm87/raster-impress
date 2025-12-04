import sys
import argparse
import logging
from pathlib import Path
from raster_impress import __version__
from raster_impress.raster_analysis import (
    analyze_raster,
    compute_histogram,
    compute_ndvi,
    compute_slope,
    compute_hillshade_auto,
    print_metadata,
    quality_check,
)

# Logging setup
logger = logging.getLogger("raster-impress")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Raster analysis tool with automatic TIF and plot generation"
    )
    parser.add_argument("filepath", help="Path to input raster file")
    parser.add_argument("--stats", action="store_true", help="Compute basic statistics")
    parser.add_argument(
        "--histogram",
        nargs="?",
        const=None,
        help="Compute histogram. Optional output filename; default adds '_hist.tif' to input raster",
    )
    parser.add_argument(
        "--ndvi",
        nargs="?",
        const=None,
        help="Compute NDVI (requires at least 2 bands). Optional output filename; default adds '_ndvi.tif' to input raster",
    )
    parser.add_argument(
        "--slope",
        nargs="?",
        const=None,
        help="Compute slope (DEM required). Optional output filename; default adds '_slope.tif' to input raster",
    )
    parser.add_argument(
        "--hillshade",
        nargs="?",
        const=None,
        help="Compute hillshade (DEM required). Optional output filename; default adds '_hillshade.tif' to input raster",
    )
    parser.add_argument("--metadata", action="store_true", help="Show raster metadata")
    parser.add_argument("--quality", action="store_true", help="Perform quality check")
    parser.add_argument("--silent", action="store_true", help="Suppress log output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()
    input_path = Path(args.filepath)

    if args.silent:
        logger.setLevel(logging.WARNING)

    results = {}

    # ----------------------------
    # Stats
    # ----------------------------
    if args.stats:
        results["stats"] = analyze_raster(str(input_path), verbose=not args.silent)

    # ----------------------------
    # Histogram
    # ----------------------------
    if args.histogram is not None or "--histogram" in sys.argv:
        hist_file = args.histogram or f"{input_path.stem}_hist.tif"
        results["histogram"] = compute_histogram(
            str(input_path), output_tif=hist_file, verbose=not args.silent
        )

    # ----------------------------
    # NDVI
    # ----------------------------
    if args.ndvi is not None or "--ndvi" in sys.argv:
        ndvi_file = args.ndvi or f"{input_path.stem}_ndvi.tif"
        results["ndvi"] = compute_ndvi(
            str(input_path), output_tif=ndvi_file, verbose=not args.silent
        )

    # ----------------------------
    # Slope
    # ----------------------------
    if args.slope is not None or "--slope" in sys.argv:
        slope_file = args.slope or f"{input_path.stem}_slope.tif"
        results["slope"] = compute_slope(
            str(input_path), output_tif=slope_file, verbose=not args.silent
        )

    # ----------------------------
    # Hillshade
    # ----------------------------
    if args.hillshade is not None or "--hillshade" in sys.argv:
        hill_file = args.hillshade or f"{input_path.stem}_hillshade.tif"
        results["hillshade"] = compute_hillshade_auto(
            str(input_path), output_tif=hill_file, verbose=not args.silent
        )

    # ----------------------------
    # Metadata
    # ----------------------------
    if args.metadata:
        results["metadata"] = print_metadata(str(input_path), verbose=not args.silent)

    # ----------------------------
    # Quality check
    # ----------------------------
    if args.quality:
        results["quality"] = quality_check(str(input_path), verbose=not args.silent)

    # ----------------------------
    # Print results
    # ----------------------------
    for key, value in results.items():
        logger.info("=== %s ===", key.upper())
        if isinstance(value, dict):
            for k, v in value.items():
                logger.info("%s: %s", k, v)
        elif isinstance(value, list):
            logger.info("%s", value)
        else:
            logger.info("%s", value)

    return results


if __name__ == "__main__":
    sys.exit(main() or 0)
