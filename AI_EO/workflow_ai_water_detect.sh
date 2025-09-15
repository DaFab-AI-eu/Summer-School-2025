#!/bin/bash
set -euo pipefail

# Input parameters
TILE_ID="${1:?Tile ID required}"
CACHE_FOLDER="${2:-./data/cache}"
TEMPORARY_FOLDER="${3:-./data/temporary}"
MODEL="${4:-./model_v1.0.0.onnx}"
SUB_NAME="${5:-prithvi}"
PATCH_COUNT="${5:-2}"
BATCH_SIZE="${6:-1}"
# GPU on by default, pass “--no-gpu” as 4th arg to disable
GPU_FLAG="--gpu"
if [[ "${7:-}" == "--no-gpu" ]]; then
  GPU_FLAG=""
fi

# Fixed parameters for tile splitting and processing
# PATCH_COUNT = 2 # -1
RESOLUTION=10
PATCH_WIDTH=896
PATCH_HEIGHT=896
PATCH_OVERLAP=150
#BATCH_SIZE=1

# Specify paths
METADATA_JSON="$CACHE_FOLDER/json/$TILE_ID.json"
ASSETS_FOLDER="$CACHE_FOLDER/assets/$TILE_ID"
VISUAL_FOLDER="$CACHE_FOLDER/visuals/$TILE_ID"
PATCHES_FOLDER="$TEMPORARY_FOLDER/patches/$TILE_ID"
PROCESS_FOLDER="$TEMPORARY_FOLDER/process/$TILE_ID"
COMBINED_FOLDER="$TEMPORARY_FOLDER/final/$TILE_ID"
CONTOURS_FOLDER="$TEMPORARY_FOLDER/contours/$TILE_ID"
VRT_FILE="$CONTOURS_FOLDER/temp.vrt"

mkdir -p "$CACHE_FOLDER/json" "$ASSETS_FOLDER" "$PATCHES_FOLDER" "$PROCESS_FOLDER" "$CONTOURS_FOLDER" "$COMBINED_FOLDER" "$VISUAL_FOLDER"

# Download metadata if not present
if [[ ! -f "$METADATA_JSON" ]]; then
  echo "Fetching metadata for $TILE_ID..."
  if ! curl -f "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/$TILE_ID" -o "$METADATA_JSON"; then
    echo "Tile $TILE_ID not found." >&2
    exit 1
  fi
fi


# Download different image bands in parallel
echo "Downloading bands..."
bands="blue red green nir08 nir swir16 swir22 scl"

export ASSETS_FOLDER RESOLUTION METADATA_JSON
parallel -j5 '
  target="$ASSETS_FOLDER/{}_$RESOLUTION.nc"
  if [[ -f "$target" ]]; then
    echo "$target already exists."
  else
    echo "Downloading $target..."
    python ./scripts/t-fetch-s2-tile.py -o "$target" -b {} -p \
    -r "$RESOLUTION" "$METADATA_JSON"
  fi
' ::: $bands


# Generate visual
VISUAL_FILE="$VISUAL_FOLDER/rgb_$TILE_ID.tiff"
echo "Generating visual at $VISUAL_FILE..."
python ./scripts/t-generate-visual.py -o "$VISUAL_FILE" rgb \
  -r "$ASSETS_FOLDER/red_10.nc" \
  -g "$ASSETS_FOLDER/green_10.nc" \
  -b "$ASSETS_FOLDER/blue_10.nc"

# "SWIR1,NIR,RED Composite"
VISUAL_FILE="$VISUAL_FOLDER/swir_nir_red_$TILE_ID.tiff"
echo "Generating visual at $VISUAL_FILE..."
python ./scripts/t-generate-visual.py -o "$VISUAL_FILE" rgb \
  -r "$ASSETS_FOLDER/swir16_10.nc" \
  -g "$ASSETS_FOLDER/nir_10.nc" \
  -b "$ASSETS_FOLDER/red_10.nc"

VISUAL_FILE="$VISUAL_FOLDER/scl_$TILE_ID.tiff"
echo "Generate visual at $VISUAL_FILE... "
python ./scripts/t-generate-visual.py \
    --output $VISUAL_FILE scl "$ASSETS_FOLDER/scl_10.nc"


# Split tile into patches
echo "Splitting tile into patches..."
python ./scripts/t-split-s2-tile.py \
  --width "$PATCH_WIDTH" --height "$PATCH_HEIGHT" --overlap "$PATCH_OVERLAP" \
  -c $PATCH_COUNT \
  --output "$PATCHES_FOLDER" "$ASSETS_FOLDER"/*.nc

# Filter valid patches
echo "Filtering patches..."
VALID_PATCHES_JSON=$(python ./scripts/t-filter-enumerate-patches.py --filter \
--max-bad-pixels 0.99 --max-cloud-pixels 1 --min-water-pixels 1e-5 "$PATCHES_FOLDER"/*)
VALID_PATCHES=($(echo "$VALID_PATCHES_JSON" | jq -r '.[0].patches | split(" ")[]'))
echo "Found ${#VALID_PATCHES[@]} valid patches."

# compute tile global mean std for bands (used in AI inferencing data preprocessing)
# mean, std constant for any tile for Prithvi - https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
# Prithvi has means, stds https://github.com/IBM/terratorch/blob/d582857b7ae76f5ccd0ad9d9ebfb562582deebca/terratorch/models/backbones/prithvi_vit.py#L30C1-L31C66
# bands for L2A S2 (without CIRRUS band in L1C - [1,2,3,8,10,11] instead of [1,2,3,8,11,12]) https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_register.py
$STATS_PATH = "./stats/global_stats_s2l2a_${SUB_NAME}.json" # global_stats_s2l2a_terramind

# Run AI inference on patches
echo "Processing patches..."
python ./scripts/g-process-s2-patch.py $GPU_FLAG \
  --stats $STATS_PATH --dtype "float32" \
  --model "$MODEL" --batch-size "$BATCH_SIZE" --output "$PROCESS_FOLDER" \
  "${VALID_PATCHES[@]}"

echo "Building VRT file..."
python ./scripts/t-build-vrt-file.py --output "$VRT_FILE" "$PROCESS_FOLDER"/*.tif

# combine patches/sub-tiles (.tif) into one tile (.tif)
$COMBINED_FILE = Join-Path $CACHE_FOLDER "observed_water_mask_${SUB_NAME}_v1.tif"
echo "Generate $COMBINED_FILE... "
python ./scripts/t-raster-translate.py --output $COMBINED_FILE -m $METADATA_JSON \
 $VRT_FILE -b "" -f