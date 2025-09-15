# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
import json
import logging
from pathlib import Path
from typing import Any, TypeAlias, cast

# on Windows, there is an issue when importing pandas/geopandas before GDAL, so this
# should be kept here
from osgeo import gdal  # isort: skip # pyright: ignore[reportMissingTypeStubs]

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from numpy.typing import NDArray
from odc.geo.geobox import GeoBox
from shapely import Polygon

FloatArray: TypeAlias = NDArray[np.floating[Any]]

LOGGER = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="raster translate from GDAL VRT file to TIFF file "
        "(combine patches into one tile)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path to the .tif file to generate",
    )
    parser.add_argument(
        "-b",
        "--output-band",
        type=str,
        required=False,
        default="observed_water_mask",
        help="band name to save in generated NetCDF file",
    )
    parser.add_argument(
        "-m", "--metadata-path", type=Path, help="path to S2 tile metadata"
    )
    parser.add_argument(
        "-f",
        "--flag-resolution",
        action="store_true",
        default=False,
        help="if provided option by resolution, not by output size",
    )
    parser.add_argument("-r", "--resolution", type=int, required=False, default=10)
    parser.add_argument(
        "vrt_path",
        type=Path,
        help="path to the GDAL VRT file to gdal_translate",
    )

    args = parser.parse_args()

    output_path: Path = args.output
    vrt_path: Path = args.vrt_path
    col_name: str = args.output_band
    metadata_path: Path = args.metadata_path
    resolution: int = args.resolution
    flag_resolution: bool = args.flag_resolution

    gdal.UseExceptions()

    # combine patches (.tif) into one tile (.tif)
    # TODO fix size (in case missing patches?) output with no values 0 ?
    vrt_ds = gdal.Open(vrt_path)  # type: ignore
    LOGGER.info(f"translate to file {output_path}")

    with open(metadata_path, "r") as fp:
        boundaries = gpd.GeoDataFrame.from_features([json.load(fp)], crs="epsg:4326")  # pyright: ignore[reportUnknownMemberType]

    epsg = (
        cast(str, boundaries["proj:code"][0])
        if "proj:code" in boundaries.columns
        else f"EPSG:{boundaries['proj:epsg'][0]}"
    )
    # if missing values (no_data) is present in S2 -> S2 geometry not the whole tile_grid -> partial -> problems with xr.merge !!!
    boundaries = boundaries.to_crs(epsg)
    tile_bounds = cast(
        Polygon, boundaries.geometry.iloc[0]
    ).bounds  # projected (in meters) bounds of the tile-grid
    geobox = GeoBox.from_bbox(bbox=tile_bounds, resolution=resolution, crs=epsg)

    translate_params: dict[str, int] = {}
    # computes output_size based input_size * resolution_out / resolution_in
    if flag_resolution:
        translate_params = {
            "xRes": resolution,
            "yRes": resolution,
        }
    # computes resolution resolution_in * from output_size / input_size
    else:
        translate_params = {
            "width": geobox.width,
            "height": geobox.height,
        }

    gdal.Translate(
        output_path,
        vrt_ds,
        options=gdal.TranslateOptions(
            **translate_params,  # type: ignore
            # xRes=resolution, yRes=resolution, # instead of width, height TODO check
        ),  # change width, height
    )  # scale 4*4 converted .tiff bigger file ~8G because of resolution

    if col_name:
        # convert TIF to netCDF
        tiff_ds = rioxarray.open_rasterio(output_path)
        tiff_ds = cast(xr.Dataset, tiff_ds.rename({"band": "time"}))  # if several bands -> in time dim
        tiff_ds = tiff_ds.to_dataset(name=col_name)
        tiff_ds[[col_name]].to_netcdf(output_path.with_suffix(".nc"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
