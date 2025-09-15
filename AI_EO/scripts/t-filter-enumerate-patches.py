import argparse
import itertools as it
import json
import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


class ClassificationMask(IntEnum):
    NO_DATA = 0
    SATURATED_OR_DEFECTIVE = 1
    CASTED_SHADOWS = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_OR_ICE = 11


DEFAULT_BAD_PIXELS_MASKS = (
    ClassificationMask.NO_DATA,
    ClassificationMask.SATURATED_OR_DEFECTIVE,
    ClassificationMask.CLOUD_HIGH_PROBABILITY,
    ClassificationMask.SNOW_OR_ICE,
)


def check_patch(
    patch_patch: Path,
    bad_pixel_classes: Sequence[ClassificationMask],
    max_bad_pixels: float,
    min_vegetation_pixels: float,
    min_water_pixels: float,
    max_cloud_pixels: float,
) -> bool:
    patch = xr.open_dataset(patch_patch)  # pyright: ignore[reportUnknownMemberType]

    # scene classification data
    scl_data = cast(NDArray[np.uint8], patch["scl"].values)  # pyright: ignore[reportUnknownMemberType]

    n_pixels = cast(int, np.prod(scl_data.shape))
    bad_pixels = np.isin(scl_data, bad_pixel_classes).sum()
    vegetation_pixels = np.equal(scl_data, ClassificationMask.VEGETATION).sum()
    water_pixels = np.equal(scl_data, ClassificationMask.WATER).sum()
    cloudy_pixels = np.isin(
        scl_data,
        [
            ClassificationMask.CLOUD_MEDIUM_PROBABILITY,
            ClassificationMask.CLOUD_HIGH_PROBABILITY,
        ],
    ).sum()

    # convert to percentages
    bad_pixels = bad_pixels / n_pixels
    vegetation_pixels = vegetation_pixels / n_pixels
    water_pixels = water_pixels / n_pixels
    cloudy_pixels = cloudy_pixels / n_pixels

    return cloudy_pixels <= max_cloud_pixels and (
        bad_pixels <= max_bad_pixels
        or (
            vegetation_pixels >= min_vegetation_pixels
            and water_pixels >= min_water_pixels
        )
    )


def glob_paths(patterns: list[str]) -> Iterable[Path]:
    import glob

    return (Path(filepath) for pattern in patterns for filepath in glob.glob(pattern))


def parse_bad_pixel_classes(value: str) -> set[ClassificationMask]:
    if not value:
        return set()

    parts = [p.strip() for p in value.strip().split(",")]

    name_map = {
        k.lower().replace("_", "-"): v for k, v in ClassificationMask.__members__.items()
    }

    classes: set[ClassificationMask] = set()
    for name in parts:
        if name not in name_map:
            raise ValueError(f"classification mask '{name}' does not exist")
        classes.add(name_map[name])
    return classes



def main():
    parser = argparse.ArgumentParser(description="split tile into patches")

    parser.add_argument(
        "-c",
        "--count",
        type=int,
        required=False,
        default=None,
        help="maximum number of patches to enumerate",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="number of workers to enumerate patches for",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help="whether to filter patches or not",
    )
    parser.add_argument(
        "-b",
        "--bad-pixel-classes",
        type=parse_bad_pixel_classes,
        required=False,
        default=set(DEFAULT_BAD_PIXELS_MASKS),
        help="pixel classes considered as bad pixels",
    )
    parser.add_argument(
        "--extra-bad-pixel-classes",
        type=parse_bad_pixel_classes,
        required=False,
        default=set(),
        help="extra pixel classes considered as bad pixels",
    )
    parser.add_argument(
        "--max-bad-pixels",
        type=float,
        required=False,
        default=0.7,
        help="maximum percentage of bad pixels to include a patch (only when filtering)",
    )
    parser.add_argument(
        "--max-cloud-pixels",
        type=float,
        required=False,
        default=0.30,
        help="maximum percentage of cloud pixels to include a patch (only when filtering)",
    )
    parser.add_argument(
        "--min-vegetation-pixels",
        type=float,
        required=False,
        default=0,
        help="minimum percentage of vegetation pixels to include a patch (only when filtering)",
    )
    parser.add_argument(
        "--min-water-pixels",
        type=float,
        required=False,
        default=0,
        help="minimum percentage of water pixels to include a patch (only when filtering)",
    )
    parser.add_argument(
        "patches", type=str, nargs="+", help="patches (folder) to filter and enumerate"
    )

    args = parser.parse_args()

    patch_globs: list[str] = args.patches
    workers: int = args.workers
    count: int | None = args.count
    filtering: bool = args.filter
    bad_pixel_classes: set[ClassificationMask] = args.bad_pixel_classes
    extra_bad_pixel_classes: set[ClassificationMask] = args.extra_bad_pixel_classes
    max_bad_pixels: float = args.max_bad_pixels
    max_cloud_pixels: float = args.max_cloud_pixels
    min_vegetation_pixels: float = args.min_vegetation_pixels
    min_water_pixels: float = args.min_water_pixels

    bad_pixel_classes |= extra_bad_pixel_classes
    LOGGER.info(f"considering {bad_pixel_classes} as bad pixels")

    paths = list(glob_paths(patch_globs))

    if filtering:
        paths = list(
            it.islice(
                filter(
                    lambda p: check_patch(
                        p,
                        bad_pixel_classes=list(bad_pixel_classes),
                        max_bad_pixels=max_bad_pixels,
                        max_cloud_pixels=max_cloud_pixels,
                        min_vegetation_pixels=min_vegetation_pixels,
                        min_water_pixels=min_water_pixels
                    ),
                    paths,
                ),
                count,
            )
        )

    json.dump(
        [
            {"patches": " ".join(path.as_posix() for path in batch_paths)}
            for batch_paths in it.batched(paths, (len(paths) + workers - 1) // workers)
        ],
        sys.stdout,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    main()
