import argparse
import itertools as it
import logging
from pathlib import Path
from typing import Iterable

import xarray as xr

LOGGER = logging.getLogger(__name__)


def split_save2eopatch(
    *,
    dataset: xr.Dataset,
    patches_folder: Path,
    height: int,
    width: int,
    overlap: int = 0,
    numbers_of_split: int | None = None,
    begin_x: int = 0,
    begin_y: int = 0,
):
    if dataset.rio.crs.units_factor != ("metre", 1):
        raise ValueError(
            f"Dataset CRS should have axis units in metres, found {dataset.rio.crs.units_factor}."
        )

    W, H = len(dataset.x), len(dataset.y)

    x_rng = range(begin_x, W - width + 1, width - overlap)
    y_rng = range(begin_y, H - height + 1, height - overlap)

    xy_index = list(it.product(x_rng, y_rng))

    if (W - width) % (width - overlap) != 0:
        xy_index.extend((W - width, y) for y in y_rng)

    if (H - height) % (height - overlap) != 0:
        xy_index.extend((x, H - height) for x in x_rng)

    # bottom right corner if needed
    if (W - width) % (width - overlap) != 0 and (H - height) % (height - overlap) != 0:
        xy_index.append((W - width, H - height))

    xy_index = sorted(xy_index)

    # TODO padding from all 4 sides of the tile
    for num_split, (ind_x, ind_y) in enumerate(xy_index):
        if numbers_of_split == num_split:
            break

        LOGGER.info(f"generating split {num_split} at ({ind_x}, {ind_y})...")

        # create patch
        patch = dataset.isel(
            x=slice(ind_x, ind_x + width),
            y=slice(ind_y, ind_y + height),
        )

        # not sure why this is needed?
        patch = (
            patch.rio.write_crs(dataset.rio.crs)
            .rio.set_spatial_dims(x_dim=dataset.rio.x_dim, y_dim=dataset.rio.y_dim)
            .rio.write_coordinate_system()
        )

        # save patch
        patch.to_netcdf(  # type: ignore
            patches_folder.joinpath(f"eopatch_x_{ind_x}_y_{ind_y}_{num_split}.nc")
        )


def glob_paths(patterns: list[str]) -> Iterable[Path]:
    import glob

    return (Path(filepath) for pattern in patterns for filepath in glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="split tile into patches")

    parser.add_argument(
        "-o", "--output", type=Path, help="output folder that will contain the patches"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        required=False,
        default=None,
        help="number of patches to generate",
    )
    parser.add_argument(
        "--overlap", type=int, default=0, help="overlap between patches"
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="width of the patches to generate",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="height of the patches to generate",
    )
    parser.add_argument(
        "tile_paths", type=str, nargs="+", help="path to the tile to split (nc file)"
    )

    args = parser.parse_args()

    tile_paths: list[str] = args.tile_paths
    eopatches_path: Path = args.output

    count: int | None = args.count
    overlap: int = args.overlap
    width: int = args.width
    height: int = args.height

    datasets: list[xr.Dataset] = []
    for filepath in glob_paths(tile_paths):
        dataset = xr.open_dataset(filepath, decode_coords="all")  # type: ignore
        LOGGER.info(f"loaded dataset from {filepath}: {dataset}")

        datasets.append(dataset)

    dataset = xr.merge(  # type: ignore
        datasets, compat="override", join="override", combine_attrs="override"
    )

    del datasets

    # force load data into memory for faster computation
    dataset = dataset.load()  # type: ignore

    LOGGER.info(f"dataset loaded: {dataset}")

    split_save2eopatch(
        dataset=dataset,
        patches_folder=eopatches_path,
        height=height,
        width=width,
        overlap=overlap,
        numbers_of_split=count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
