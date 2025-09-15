import argparse
import itertools as it
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Literal, TypeAlias, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
import rasterio
import xarray as xr
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

FloatArray: TypeAlias = NDArray[np.floating[Any]]
Band: TypeAlias = Literal["blue", "green", "red", "nir", "nir08", "swir16", "swir22"]


def normalize_meanstd(
    data: FloatArray,
    nodata_value: float = 0,
    mean: FloatArray | None = None,
    std: FloatArray | None = None,
) -> FloatArray:
    """
    Normalize each band of the input array by subtracting the nonzero mean and
    dividing by the nonzero standard deviation then fill nodata values with 0.
    """

    # taken from omnimask
    mask = data != nodata_value
    data = data.astype(float)
    data[~mask] = np.nan

    if mean is None or std is None:
        # mean of band, maybe batch too add in axis
        mean = cast(FloatArray, np.nanmean(data, axis=(-1, -2), keepdims=True))
        std = cast(FloatArray, np.nanstd(data, axis=(-1, -2), keepdims=True))

        # prevent division by zero
        std = np.where(std == 0, 1, std)
    else:
        mean = mean.astype(float)[None, :, None, None]
        std = std.astype(float)[None, :, None, None]

    data = (data - mean) / std

    # fill original nodata values with 0
    data[~mask] = 0

    return data


def softmax(x: FloatArray, axis: int | None = None):
    # https://github.com/scipy/scipy/blob/v1.16.0/scipy/special/_logsumexp.py#L257-L353
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def predict(
    model: ort.InferenceSession,
    input: FloatArray,
    mean: FloatArray | None = None,
    std: FloatArray | None = None,
    dtype_save: str = "float32",
) -> tuple[FloatArray]:
    """
    Simple method to call model predict with normalization, but typed.
    """
    input_data = normalize_meanstd(input, mean=mean, std=std)
    input_data = input_data.astype(dtype_save) # for "float16" only prithvi
    return cast(
        tuple[FloatArray],
        model.run(None, {"input": input_data}),  # pyright: ignore[reportUnknownMemberType]
    )


def crop_array(array: FloatArray, buffer: int) -> FloatArray:
    """
    Crop height and width of a 4D array given a buffer size. Array has shape B x H x W x C
    """
    assert array.ndim == 4, "input array is of wrong dimension, needs to be 4D: BHWC"
    if not buffer:  # for zero buffer
        return array
    return array[:, buffer:-buffer:, buffer:-buffer:, :]


def run_prediction(
    patch_paths: list[Path],
    model: ort.InferenceSession,
    pad_buffer: int = 12,
    crop_buffer: int = 12,
    mean: FloatArray | None = None,
    std: FloatArray | None = None,
    dtype_save: str = "float32",
    save_all: bool = False,
    bands_order: tuple[Band, ...] = ("red", "green", "nir08"),
) -> list[xr.Dataset]:
    # https://github.com/sentinel-hub/field-delineation/blob/main/fd/prediction.py

    # load all patches
    patches = [
        xr.open_dataset(patch_folder, decode_coords="all")  # pyright: ignore[reportUnknownMemberType]
        for patch_folder in patch_paths
    ]

    # load data
    bands: list[FloatArray] = []
    for folder, patch in zip(patch_paths, patches, strict=True):
        try:
            bands.append(
                np.stack(
                    [
                        cast(FloatArray, patch[band].values)  # pyright: ignore[reportUnknownMemberType]
                        for band in bands_order
                    ],
                    axis=-1,
                )
            )
        except Exception as err:
            raise RuntimeError(f"failed to load dataset from {folder}") from err

    data: FloatArray = np.pad(
        np.concatenate(bands, axis=0),
        [(0, 0), (pad_buffer, pad_buffer), (pad_buffer, pad_buffer), (0, 0)],
        mode="edge",
    )
    # data = np.permute_dims(
    #     data, (0, 3, 1, 2)
    # )  # (batch, H, W, channels) -> (batch, channels, H, W)
    data = data.transpose((0, 3, 1, 2))
    predicted = predict(
        model, data, mean=mean, std=std, dtype_save=dtype_save
    )  # data bands after batch transpose (input should be in format (bands (red,green,NIR), height, width))

    #out_val = np.permute_dims(predicted[0], (0, 2, 3, 1))
    out_val = predicted[0].transpose((0, 2, 3, 1))
    out_val = softmax(out_val, axis=-1)
    predicted_cropped = {
        "output": crop_array(out_val, buffer=crop_buffer),
    }

    patches_out = []
    for i_patch, patch in enumerate(patches):
        for name, value in predicted_cropped.items():
            idxs = range(0, value.shape[-1]) if save_all else [1] # sum (prob_i) = 1 -> one prob_i is redundant
            for idx in idxs:  # number of classes in the last dim
                patch[f"predicted_{name}_{idx}"] = (
                    ("time", "y", "x"),
                    value[i_patch : i_patch + 1, ..., idx].astype("float32"),
                )  # here shape several classes as an output probability map
            patch = patch[[f"predicted_{name}_{idx}" for idx in idxs]] # crs save
            patches_out.append(patch)

    return patches_out


def glob_paths(patterns: list[str]) -> Iterable[Path]:
    import glob

    return (Path(filepath) for pattern in patterns for filepath in glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="process patch for cloud detection")

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output folder where processed patches will be stored",
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="use GPU for inference"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="path to the .onnx containing the model",
    )
    parser.add_argument(
        "-t",
        "--stats",
        type=Path,
        default=None,
        help="path to the .json containing mean, std",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="batch size to use"
    )
    parser.add_argument(
        "-s", "--skip-existing", action="store_true", help="skip existing files"
    )
    parser.add_argument(
        "-p",
        "--save-all",
        action="store_true",
        help="if true save all predicted outputs, false - save only one (index = 1)",
    )
    parser.add_argument(
        "-d", "--dtype", type=str, default="float32", help="dtype to use"
    )
    parser.add_argument(
        "-l", "--bands", type=str, nargs="+", default=(
            "blue",
            "green",
            "red",
            "nir08",
            "swir16",
            "swir22",
        ), help="bands in S2 to use"
    )
    parser.add_argument(
        "patches", type=str, nargs="+", help="path to the folders containing the patch"
    )

    args = parser.parse_args()

    patch_patterns: list[str] = args.patches
    model_path: Path = args.model
    with_gpu: bool = args.gpu
    batch_size: int = args.batch_size
    output_folder: Path = args.output
    skip_existing: bool = args.skip_existing
    stats_path: Path = args.stats
    save_all: bool = args.save_all
    dtype_save: str = args.dtype
    bands_order: tuple[Band, ...] = args.bands # ("red", "green", "nir08")

    means, stds = None, None
    if stats_path:
        with open(stats_path, "r") as file:
            stats = json.load(file)
        means = np.array([stats["mean"][band] for band in bands_order])
        stds = np.array([stats["std"][band] for band in bands_order])

    LOGGER.info(f"mean & std for preprocessing of patches = {means}, {stds}")

    model = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider" if with_gpu else "CPUExecutionProvider"],
    )

    patch_folders = list(glob_paths(patch_patterns))

    if skip_existing:
        patch_folders = [
            folder
            for folder in patch_folders
            if not output_folder.joinpath(folder.name).exists()
        ]

    for batch_patch_folders in it.batched(patch_folders, batch_size):
        name = ", ".join(p.name for p in batch_patch_folders)
        LOGGER.info(f"running predict on {name}... ")

        processed_patches = run_prediction(
            list(batch_patch_folders),
            model,
            pad_buffer=0,
            crop_buffer=0,
            mean=means,
            std=stds,
            dtype_save=dtype_save,
            bands_order=bands_order,
            save_all=save_all,
        )  # TODO add in args - pad_buffer=0, crop_buffer=0

        for patch_folder, processed_patch in zip(
            batch_patch_folders, processed_patches, strict=True
        ):
            processed_patch.isel(time=0).rio.to_raster(output_folder.joinpath(
                      f"{patch_folder.stem}-{processed_patch.rio.crs.to_epsg()}.tiff"
                  ))

            # processed_patch.to_netcdf(output_folder.joinpath(patch_folder.name))  # type: ignore
          #   combined_list = [np.nan_to_num(processed_patch[var_name].isel(time=0).values, 0) for var_name in processed_patch]
          #   height, width = combined_list[0].shape
          #   with (
          #     open(
          #         output_folder.joinpath(
          #             f"{patch_folder.stem}-{processed_patch.rio.crs.to_epsg()}.tiff"
          #         ),
          #         "wb",
          #     ) as fp,
          #     rasterio.open(  # pyright: ignore[reportUnknownMemberType]
          #         fp,
          #         "w",
          #         driver="GTiff",
          #         width=width,
          #         height=height,
          #         count=len(combined_list),  # number of bands to save
          #         dtype="float32",
          #         transform=rasterio.transform.from_bounds(  # pyright: ignore[reportUnknownMemberType]
          #             *processed_patch.rio.bounds(), width=width, height=height
          #         ),
          #         crs=processed_patch.rio.crs,
          #     ) as rio_fp,  # type: ignore
          # ):
          #     for idx, combined in enumerate(combined_list):
          #         rio_fp.write(combined, indexes=idx + 1)  # type: ignore


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    LOGGER.setLevel(logging.INFO)
    main()
