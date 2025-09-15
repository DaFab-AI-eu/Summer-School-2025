import argparse
import logging
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def parse_rgb(value: str) -> tuple[int, int, int]:
    if value.startswith("#"):
        value = value[1:]
    return (
        int(value[0:2], base=16),
        int(value[2:4], base=16),
        int(value[4:6], base=16),
    )


def get_color_map_scl() -> dict[int, Sequence[int]]:
    return {
        0: [0, 0, 0],  # No Data (Missing data) - black
        1: [255, 0, 0],  # Saturated or defective pixel - red
        2: [47, 47, 47],
        # Topographic casted shadows ("Dark features/Shadows" for data before 2022-01-25) - very dark grey
        3: [100, 50, 0],  # Cloud shadows - dark brown
        4: [0, 160, 0],  # Vegetation - green
        5: [255, 230, 90],  # Not-vegetated - dark yellow
        6: [0, 0, 255],  # Water (dark and bright) - blue
        7: [128, 128, 128],  # Unclassified - dark grey
        8: [192, 192, 192],  # Cloud medium probability - grey
        9: [255, 255, 255],  # Cloud high probability - white
        10: [100, 200, 255],  # Thin cirrus - very bright blue
        11: [255, 150, 255],  # Snow or ice - very bright pink
    }


def get_color_map_esa_world_cover(path2meta: Path) -> dict[int, Sequence[int]]:
    import json

    with open(path2meta, "r") as file:
        items_dict = json.load(file)

    color_hint: list[dict[str, Any]] = items_dict["features"][0]["assets"]["map"][
        "classification:classes"
    ]

    return {
        cast(int, item["value"]): parse_rgb(item["color-hint"]) for item in color_hint
    }


def get_color_map_final_water() -> dict[int, Sequence[int]]:
    return {
        0: [0, 0, 0],  # no water
        1: [0, 0, 255],  # permanent
        2: [0, 100, 255],  # seasonal
        3: [80, 200, 200],  # seasonal observed
        4: [0, 255, 0],  # flood observed
        5: [255, 0, 0],  # drought, less water in observed then permanent
        6: [100, 100, 100],  # no data reference
        7: [50, 50, 50],  # no data observed
    }


def visualize_rgb(args: argparse.Namespace):
    red_path: Path = args.red
    blue_path: Path = args.blue
    green_path: Path = args.green

    output: Path = args.output
    scale: float = args.scale

    dataset = xr.merge(  # type: ignore
        [
            xr.open_dataset(path, decode_coords="all")  # type: ignore
            for path in (red_path, green_path, blue_path)
        ]
    )

    for band in ("red", "green", "blue"):
        dataset[band] = dataset[band].astype("float32") * scale  # type: ignore
        dataset[band] = dataset[band].clip(0, 1)

    dataset[["red", "green", "blue"]].isel(time=0).rio.to_raster(output)


def generate_visual_dataset(
    data: xr.DataArray,
    color_map: dict[int, Sequence[int]],
):
    LOGGER.info("building visual dataset...")

    arr_map = np.zeros((max(color_map.keys()) + 1, 3), dtype=np.uint8)
    for i, c in color_map.items():
        arr_map[i, :] = c

    values = arr_map[data.values]

    return xr.Dataset(
        {
            band: (("y", "x"), values[..., idx])
            for idx, band in enumerate(("red", "green", "blue"))
        },
        coords=data.coords, # type: ignore
    )


def visualize_scl(args: argparse.Namespace):
    patch_path: Path = args.patch
    output: Path = args.output

    dataset = xr.open_dataset(patch_path, decode_coords="all")  # type: ignore
    rgb_dataset = generate_visual_dataset(
        data=dataset["scl"].isel(time=0), color_map=get_color_map_scl()
    )

    LOGGER.info(f"generating {output}...")
    rgb_dataset.rio.to_raster(output)


def visualize_esa(args: argparse.Namespace):
    patch_path: Path = args.patch
    metadata_path: Path = args.metadata
    output: Path = args.output

    dataset = xr.open_dataset(patch_path, decode_coords="all")  # type: ignore
    rgb_dataset = generate_visual_dataset(
        data=dataset["map"].isel(time=0),
        color_map=get_color_map_esa_world_cover(metadata_path),
    )

    LOGGER.info(f"generating {output}...")
    rgb_dataset.rio.to_raster(output)


def visualize_final_water(args: argparse.Namespace):
    patch_path: Path = args.patch
    output: Path = args.output

    dataset = xr.open_dataset(patch_path, decode_coords="all")  # type: ignore
    rgb_dataset = generate_visual_dataset(
        data=dataset["final_mask"].isel(time=0),
        color_map=get_color_map_final_water(),
    )

    LOGGER.info(f"generating {output}...")
    rgb_dataset.rio.to_raster(output)


def main():
    parser = argparse.ArgumentParser(
        description="convert a S2 tile dataset to an RGB image"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file",
    )

    subparsers = parser.add_subparsers(title="mode", required=True)

    rgb_parser = subparsers.add_parser("rgb")

    rgb_parser.add_argument(
        "-r",
        "--red",
        type=Path,
        required=True,
        help="file containing the red dataset",
    )

    rgb_parser.add_argument(
        "-g",
        "--green",
        type=Path,
        required=True,
        help="file containing the green dataset",
    )

    rgb_parser.add_argument(
        "-b",
        "--blue",
        type=Path,
        required=True,
        help="file containing the blue dataset",
    )
    rgb_parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=3.5 / 10000.0,
        help="scaling factor from band to rgb [0, 1] values",
    )

    rgb_parser.set_defaults(func=visualize_rgb)

    scl_parser = subparsers.add_parser("scl")
    scl_parser.add_argument(
        "patch",
        type=Path,
        help="path to the band containing scene classification to visualize",
    )
    scl_parser.set_defaults(func=visualize_scl)

    esa_world_parser = subparsers.add_parser("esa")
    esa_world_parser.add_argument("-m", "--metadata", type=Path, required=True, help="path to the metadata file for ESA World Cover")
    esa_world_parser.add_argument(
        "patch",
        type=Path,
        help="path to the band containing scene classification to visualize",
    )
    esa_world_parser.set_defaults(func=visualize_esa)

    final_water_parser = subparsers.add_parser("final-water")
    final_water_parser.add_argument(
        "patch",
        type=Path,
        help="path to the band containing scene classification to visualize",
    )
    final_water_parser.set_defaults(func=visualize_final_water)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
