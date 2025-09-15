# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

import logging
import os
from pathlib import Path
from typing import Any, Iterable, TypeAlias, cast

import numpy as np
from lxml import etree
from numpy.typing import NDArray
from osgeo import gdal  # pyright: ignore[reportMissingTypeStubs]

FloatArray: TypeAlias = NDArray[np.floating[Any]]

LOGGER = logging.getLogger(__name__)


def average_function(
    no_data: int | float = 0, round_output: bool = False, disable_warnings: bool = True
) -> str:
    """A Python function that will be added to VRT and used to calculate weighted average over overlaps

    :param no_data: no data pixel value (default = 0)
    :param round_output: flag to round the output (to 0 decimals). Useful when the final result will be in Int.
    :return: Function (as a string)
    """
    rounding = "out = np.round(out, 0)" if round_output else ""
    warnings_before = (
        "np_settings = np.seterr(all='ignore')" if disable_warnings else ""
    )
    warnings_after = "np.seterr(**np_settings)" if disable_warnings else ""
    return f"""
import numpy as np

def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize, raster_ysize, buf_radius, gt, **kwargs):
    {warnings_before}
    p, w = np.split(np.array(in_ar), 2, axis=0)
    n_overlaps = np.sum(p!={no_data}, axis=0)
    w_sum = np.sum(w, axis=0, dtype=np.float32)
    p_sum = np.sum(p, axis=0, dtype=np.float32)
    weighted = np.sum(p*w, axis=0, dtype=np.float32)
    out = np.where((n_overlaps>1) & (w_sum>0) , weighted/w_sum, p_sum/n_overlaps)
    {rounding}
    out_ar[:] = out
    {warnings_after}
"""


def write_vrt(tiff_files: list[Path], out_vrt: Path, bands: list[int] | None = None, function: str | None = None):
    """Write virtual raster

    Function that will first build a temp.vrt for the input files, and then modify it for purposes of spatial merging
    of overlaps using the provided function
    """

    if not function:
        function = average_function()

    out_vrt = out_vrt.absolute().resolve()
    tiff_files = [f.absolute().resolve() for f in tiff_files]

    # TODO: maybe making path relative to VRT in VRT file is not needed, TBC
    previous_dir = os.getcwd()
    os.chdir(out_vrt.parent)

    options_kwargs: dict[str, Any] = {}
    if bands:
        options_kwargs = {"bandList": bands}

    vrt = cast(
        gdal.Dataset,
        gdal.BuildVRT(
            out_vrt.name,
            [
                f.relative_to(out_vrt.parent, walk_up=True).as_posix()
                for f in tiff_files
            ],
            options=gdal.BuildVRTOptions(**options_kwargs)
        ),
    )
    del vrt  # flush
    os.chdir(previous_dir)

    # fix the vrt
    root = etree.parse(out_vrt).getroot()
    # TODO check if works for several bands (default average function) not for the first one
    vrt_raster_band = root.find("VRTRasterBand")  # list of bands here ???
    assert vrt_raster_band is not None

    # Add childern tags to derivedRasterBand tag
    pix_func_tag = etree.Element("PixelFunctionType")
    pix_func_tag.text = "average"

    pix_func_tag2 = etree.Element("PixelFunctionLanguage")
    pix_func_tag2.text = "Python"

    pix_func_code = etree.Element("PixelFunctionCode")
    pix_func_code.text = etree.CDATA(function)

    vrt_raster_band.insert(0, pix_func_code)
    vrt_raster_band.insert(0, pix_func_tag2)
    vrt_raster_band.insert(0, pix_func_tag)

    with open(out_vrt, "wb") as out:
        out.write(etree.tounicode(root, pretty_print=True).encode("utf-8"))  # type: ignore


def arg_int2_tuple(value: str) -> tuple[int, int]:
    try:
        p1, p2 = value.split(",")

        return (int(p1.strip()), int(p2.strip()))
    except Exception as exc:
        raise ValueError(
            f"expected two ints separated by ',', found '{value}'"
        ) from exc


def glob_paths(patterns: list[str]) -> Iterable[Path]:
    import glob

    return (Path(filepath) for pattern in patterns for filepath in glob.glob(pattern))


def parse_band_list(value: str) -> list[int]:
    return [int(c.strip()) for c in value.strip().split(",")]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="build GDAL VRT file from a set of TIFF files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="path to the GDAL VRT file to generate",
    )
    parser.add_argument(
        "-b",
        "--band-list",
        type=parse_band_list,
        required=False,
        default=None,
        help="list of bands (number separated by ',') to include in the VRT file",
    )
    parser.add_argument("tiffs_paths", type=str, nargs="+", help="path to tiff files")

    args = parser.parse_args()

    output_path: Path = args.output
    band_list: list[int] | None = args.band_list
    tiffs_paths: list[str] = args.tiffs_paths

    gdal.UseExceptions()

    output_path.parent.mkdir(exist_ok=True, parents=True)

    tiff_files = list(glob_paths(tiffs_paths))
    write_vrt(tiff_files, output_path, bands=band_list)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
