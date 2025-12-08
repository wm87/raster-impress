import argparse
import logging
import sys
from pathlib import Path

from raster_impress import __version__
from raster_impress.extractor import extract_all_features
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

# Logger
logger = logging.getLogger("raster-impress")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def main(argv=None):
    parser = argparse.ArgumentParser(description="Raster analysis & feature extraction tool")

    # Positional Raster
    parser.add_argument("raster", nargs="?", help="input raster file")

    # Analysis options
    parser.add_argument("--stats", action="store_true", help="Compute basic statistics")
    parser.add_argument("--histogram", nargs="?", const=None, help="Compute histogram")
    parser.add_argument("--ndvi", nargs="?", const=None, help="Compute NDVI")
    parser.add_argument("--slope", nargs="?", const=None, help="Compute Slope")
    parser.add_argument("--hillshade", nargs="?", const=None, help="Compute Hillshade")
    parser.add_argument("--relief", nargs="?", const=None, help="Compute synthetic Relief")
    parser.add_argument("--metadata", action="store_true", help="Show raster metadata")
    parser.add_argument("--quality", action="store_true", help="Perform quality check")

    # Feature extraction: DSM + DGM
    parser.add_argument(
        "--extract",
        nargs=2,
        metavar=("DSM", "DGM"),
        help="DSM and DGM rasters for feature extraction"
    )

    # Output / flags
    parser.add_argument("--output", help="Output folder for results")
    parser.add_argument("--silent", action="store_true", help="Suppress log output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv or sys.argv[1:])

    # Wenn kein Argument übergeben wurde, Hilfe ausgeben
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    # Silent mode
    if args.silent:
        logger.setLevel(logging.WARNING)

    # Output folder
    output_dir = Path(args.output) if args.output else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    input_path = Path(args.raster)

    # ----------------------------
    # Analysis
    # ----------------------------
    if args.stats:
        results["stats"] = analyze_raster(str(input_path), verbose=not args.silent)

    if args.histogram is not None or "--histogram" in (argv or sys.argv[1:]):
        hist_file = Path(args.histogram) if args.histogram else output_dir / f"{input_path.stem}_hist.tif"
        results["histogram"] = compute_histogram(str(input_path), output_tif=str(hist_file), verbose=not args.silent)

    if args.ndvi is not None or "--ndvi" in (argv or sys.argv[1:]):
        ndvi_file = Path(args.ndvi) if args.ndvi else output_dir / f"{input_path.stem}_ndvi.tif"
        results["ndvi"] = compute_ndvi(str(input_path), output_tif=str(ndvi_file), verbose=not args.silent)

    if args.slope is not None or "--slope" in (argv or sys.argv[1:]):
        slope_file = Path(args.slope) if args.slope else output_dir / f"{input_path.stem}_slope.tif"
        results["slope"] = compute_slope(str(input_path), output_tif=str(slope_file), verbose=not args.silent)

    if args.hillshade is not None or "--hillshade" in (argv or sys.argv[1:]):
        hill_file = Path(args.hillshade) if args.hillshade else output_dir / f"{input_path.stem}_hillshade.tif"
        results["hillshade"] = compute_hillshade(str(input_path), output_tif=str(hill_file), verbose=not args.silent)

    if args.relief is not None or "--relief" in (argv or sys.argv[1:]):
        relief_file = Path(args.relief) if args.relief else output_dir / f"{input_path.stem}_relief_plastisch.tif"
        results["relief"] = compute_relief(str(input_path), output_tif=str(relief_file), verbose=not args.silent)

    if args.metadata:
        results["metadata"] = print_metadata(str(input_path), verbose=not args.silent)

    if args.quality:
        results["quality"] = quality_check(str(input_path), verbose=not args.silent)

    # ----------------------------
    # Feature extraction
    # ----------------------------
    if args.extract:
        dsm, dgm = args.extract
        extract_results = extract_all_features(
            dop_raster=str(input_path),
            dsm_raster=str(dsm),
            dgm_raster=str(dgm),
            red_channel_raster=None,
            output_base=str(output_dir),
            verbose=not args.silent
        )
        results["extract"] = extract_results

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
