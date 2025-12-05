import argparse
import logging
import sys
from pathlib import Path

from raster_impress import __version__
from raster_impress.raster_analysis import (
    analyze_raster,
    compute_histogram,
    compute_ndvi,
    compute_slope,
    compute_hillshade,
    compute_relief,
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


def main(argv=None, output_dir=None):
    """Main function for CLI and tests.

    Args:
        argv (list[str], optional): Argument list for parsing. Defaults to None (uses sys.argv).
        output_dir (str|Path, optional): Directory where output files are stored. Defaults to current directory.
    """
    parser = argparse.ArgumentParser(
        description="Raster analysis tool with automatic TIF and plot generation"
    )
    parser.add_argument("filepath", help="Path to input raster file")
    parser.add_argument("--stats", action="store_true", help="Compute basic statistics")
    parser.add_argument("--histogram", nargs="?", const=None,
                        help="Compute histogram. Optional output filename")
    parser.add_argument("--ndvi", nargs="?", const=None,
                        help="Compute NDVI (requires at least 2 bands). Optional output filename")
    parser.add_argument("--slope", nargs="?", const=None,
                        help="Compute Slope (DEM required). Optional output filename")
    parser.add_argument("--hillshade", nargs="?", const=None,
                        help="Compute Hillshade (DEM required). Optional output filename")
    parser.add_argument("--relief", nargs="?", const=None,
                        help="Compute synthetic Relief (DEM required). Optional output filename")
    parser.add_argument("--metadata", action="store_true", help="Show raster metadata")
    parser.add_argument("--quality", action="store_true", help="Perform quality check")
    parser.add_argument("--silent", action="store_true", help="Suppress log output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv or sys.argv[1:])
    input_path = Path(args.filepath)

    if args.silent:
        logger.setLevel(logging.WARNING)

    # Output directory: CLI default or provided by tests
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ----------------------------
    # Stats
    # ----------------------------
    if args.stats:
        results["stats"] = analyze_raster(str(input_path), verbose=not args.silent)

    # ----------------------------
    # Histogram
    # ----------------------------
    if args.histogram is not None or "--histogram" in (argv or sys.argv[1:]):
        hist_file = Path(args.histogram) if args.histogram else output_dir / f"{input_path.stem}_hist.tif"
        results["histogram"] = compute_histogram(
            str(input_path), output_tif=str(hist_file), verbose=not args.silent
        )

    # ----------------------------
    # NDVI
    # ----------------------------
    if args.ndvi is not None or "--ndvi" in (argv or sys.argv[1:]):
        ndvi_file = Path(args.ndvi) if args.ndvi else output_dir / f"{input_path.stem}_ndvi.tif"
        results["ndvi"] = compute_ndvi(
            str(input_path), output_tif=str(ndvi_file), verbose=not args.silent
        )

    # ----------------------------
    # Slope
    # ----------------------------
    if args.slope is not None or "--slope" in (argv or sys.argv[1:]):
        slope_file = Path(args.slope) if args.slope else output_dir / f"{input_path.stem}_slope.tif"
        results["slope"] = compute_slope(
            str(input_path), output_tif=str(slope_file), verbose=not args.silent
        )

    # ----------------------------
    # Hillshade
    # ----------------------------
    if args.hillshade is not None or "--hillshade" in (argv or sys.argv[1:]):
        hill_file = Path(args.hillshade) if args.hillshade else output_dir / f"{input_path.stem}_hillshade.tif"
        results["hillshade"] = compute_hillshade(
            str(input_path), output_tif=str(hill_file), verbose=not args.silent
        )

    # ----------------------------
    # Relief
    # ----------------------------
    if args.relief is not None or "--relief" in (argv or sys.argv[1:]):
        relief_file = Path(args.relief) if args.relief else output_dir / f"{input_path.stem}_relief_plastisch.tif"
        results["relief"] = compute_relief(
            str(input_path), output_tif=str(relief_file), verbose=not args.silent
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
