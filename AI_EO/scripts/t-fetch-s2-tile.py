import argparse
import json
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
from odc.geo.geobox import GeoBox
from odc.stac import stac_load  # pyright: ignore[reportUnknownVariableType]
from pystac import Item
from shapely import Polygon

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=Path, required=True)
parser.add_argument("-b", "--band", type=str, required=True)
parser.add_argument("-r", "--resolution", type=int, required=True)
parser.add_argument(
    "-p",
    "--partial",
    action="store_true",
    default=False,
    help="download bands partially in case of no_data inside",
)
parser.add_argument("metadata", type=Path)

args = parser.parse_args()
output: Path = args.output
band: str = args.band
resolution: int = args.resolution
metadata_path: Path = args.metadata
partial_flag: bool = args.partial

metadata = Item.from_file(metadata_path)

params: dict[str, Any] = {}
if partial_flag:
    with open(metadata_path, "r") as fp:
        boundaries = gpd.GeoDataFrame.from_features([json.load(fp)], crs="epsg:4326")

    epsg = (
        cast(str, boundaries["proj:code"][0])
        if "proj:code" in boundaries.columns
        else f"EPSG:{boundaries['proj:epsg'][0]}"
    )

    bbox = cast(Polygon, boundaries.geometry.iloc[0]).bounds

    # if missing values (no_data) is present in S2 -> S2 geometry not the
    # whole tile_grid -> partial -> problems with xr.merge !!!
    boundaries = boundaries.to_crs(epsg)

    # projected (in meters) bounds of the tile-grid
    tile_bounds = cast(Polygon, boundaries.geometry.iloc[0]).bounds
    geobox = GeoBox.from_bbox(bbox=tile_bounds, resolution=resolution, crs=epsg)

    params = {"geobox": geobox}
else:
    params = {"resolution": resolution}

ds = stac_load([metadata], bands=[band], chunks=None, **params)
ds.to_netcdf(output)  # pyright: ignore[reportUnknownMemberType]
