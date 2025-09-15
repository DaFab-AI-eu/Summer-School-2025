Param ([string] $TileId = "S2A_46QHH_20240916_0_L2A",
    [string]$CacheFolder = ".\val",
    [string]$TemporaryFolder = ".\val",
    [string]$Model = "Prithvi-EO-V2-300M-TL-Sen1Floods11_w_t_s_896.onnx",
    [string]$SubName = "prithvi",
    [int]$PatchCount = 2,
    [int]$PatchOverlap = 100,
    [int]$PatchWidth = 224 * 4,
    [int]$BatchSize = 1,
    [switch]$DebugVisualization)
# "S2B_38NNL_20231120_0_L2A"
# S2A_46QHH_20240916_0_L2A, S2B_46QHE_20231004_0_L2A
# S2A_48PWC_20201022_0_L2A
# S2A_48QTE_20240917_0_L2A

$Resolution = 10
# $PatchCount = 2 #-1
# $PatchOverlap = 100
# $PatchWidth = 224 * 4
$PatchHeight = $PatchWidth

$CacheFolder = (Join-Path $CacheFolder $TileId)
$TemporaryFolder = (Join-Path $TemporaryFolder $TileId)

$MetaDataJson = (Join-Path $CacheFolder "json" "$TileId.json" )
$AssetsFolder = (Join-Path $CacheFolder "assets" "$TileId")
$VisualsFolder = (Join-Path $CacheFolder "visuals" "$TileId")
$PatchesFolder = (Join-Path $CacheFolder "patches" $SubName)
$ProcessFolder = (Join-Path $CacheFolder "process" $SubName)
$ContoursFolder = (Join-Path $TemporaryFolder "contour" $SubName)
$CombinedFolder = (Join-Path $TemporaryFolder "final")

@(
    $CacheFolder, $TemporaryFolder, (Split-Path -Parent $MetaDataJson), $AssetsFolder, $VisualsFolder,
    $PatchesFolder, $ProcessFolder, $ContoursFolder, $CombinedFolder
) | ForEach-Object {
    New-Item -Type Directory -Path $_ -ErrorAction Ignore | Out-Null
}


if (!(Test-Path $MetaDataJson)) {
    try {
        Invoke-RestMethod -Uri "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/$TileId" `
            -OutFile $MetadataJson
    }
    catch {
        Write-Error "Tile $TileId not found."
        Exit
    }
}

# b2 - blue
# b3 - green
# b4 - red
# b8 - nir
# b8a - nir08
# b11 - swir16
# b12 - swir22
$("red", "green", "blue", "nir", "nir08", "swir16", "swir22", "scl") | ForEach-Object -Parallel {
    $target = Join-Path $Using:AssetsFolder "${_}_$Using:Resolution.nc"

    if (Test-Path $target) {
        Write-Output "$target already exists."
    }
    else {
        Write-Output "Downloading $target..."
        #  uint16 ~ 241 M .nc assets (unit8 - 121 M .nc)
        python ".\scripts\t-fetch-s2-tile.py" -o $target -b $_ -r $Using:Resolution $Using:MetaDataJson -p
    }
}

if ($DebugVisualization) {
    $VisualFile = (Join-Path $VisualsFolder "rgb_$Resolution.tif")
    Write-Output "Generate visual for $TileId at $VisualFile... "
    python ".\scripts\t-generate-visual.py" -o $VisualFile rgb `
        --red (Join-Path $AssetsFolder "red_$Resolution.nc") `
        --green (Join-Path $AssetsFolder "green_$Resolution.nc") `
        --blue (Join-Path $AssetsFolder "blue_$Resolution.nc")

    # "SWIR1,NIR,RED Composite"
    $VisualFile = (Join-Path $VisualsFolder "swir_nir_red_$Resolution.tif")
    Write-Output "Generate visual for $TileId at $VisualFile... "
    python ".\scripts\t-generate-visual.py" -o $VisualFile rgb `
        --red (Join-Path $AssetsFolder "swir16_$Resolution.nc") `
        --green (Join-Path $AssetsFolder "nir_$Resolution.nc") `
        --blue (Join-Path $AssetsFolder "red_$Resolution.nc")

    $VisualFile = (Join-Path $VisualsFolder "scl_$Resolution.tif")
    Write-Output "Generate visual for $TileId at $VisualFile... "
    python ".\scripts\t-generate-visual.py" `
        --output $VisualFile scl (Join-Path $AssetsFolder "scl_$Resolution.nc")
}

# compute tile global mean std for bands (used in AI inferencing data preprocessing)
# mean, std constant for any tile for Prithvi - https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/download_data/convert_rgb.py
# Prithvi has means, stds https://github.com/IBM/terratorch/blob/d582857b7ae76f5ccd0ad9d9ebfb562582deebca/terratorch/models/backbones/prithvi_vit.py#L30C1-L31C66
# bands for L2A S2 (without CIRRUS band in L1C - [1,2,3,8,10,11] instead of [1,2,3,8,11,12]) https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_register.py
$StatsPath = ".\stats\global_stats_s2l2a_$SubName.json" # global_stats_s2l2a_terramind

# split tile into sub-tiles
$filepaths = @(
$("red", "green", "blue", "nir08", "swir16", "swir22", "scl") | ForEach-Object {
    Join-Path $AssetsFolder "${_}_$Resolution.nc"
})

Write-Output "filepaths to split $filepaths"
New-Item -ItemType Directory -Path $PatchesFolder -ErrorAction Ignore | Out-Null
python ".\scripts\t-split-s2-tile.py" -o $PatchesFolder -c $PatchCount  --overlap $PatchOverlap `
 --width $PatchWidth --height $PatchHeight $filepaths

# filter patches with a lot missing data & clouds
#$valid_patches = Get-ChildItem (Join-Path $PatchesFolder "*.nc")
$valid_patches = ((
    python ".\scripts\t-filter-enumerate-patches.py" `
        --filter --max-bad-pixels 0.99 --max-cloud-pixels 1 --min-water-pixels 1e-5 `
        "${PatchesFolder}/*.nc"
    ) | ConvertFrom-Json)[0].patches.Split()
$n_valid_patches = $valid_patches.Length
Write-Output "Found ${n_valid_patches} valid patches... "

# process patches
Write-Output "Process (AI inference) ${n_valid_patches} valid patches... "
python ".\scripts\g-process-s2-patch.py" --output $ProcessFolder --model $Model `
    --stats $StatsPath --dtype "float32" --batch-size $BatchSize @valid_patches

$VrtFile = (Join-Path $ContoursFolder "temp.vrt")
Write-Output "Build vrt ($VrtFile) to combine patches... "
python ".\scripts\t-build-vrt-file.py" --output $VrtFile @(Get-ChildItem $ProcessFolder)

# combine patches/sub-tiles (.tif) into one tile (.tif)
$CombinedTiff = Join-Path $CombinedFolder "observed_water_mask_${SubName}_v1.tif"
Write-Output "Generate $TileId at $CombinedTiff... "
python ".\scripts\t-raster-translate.py" --output $CombinedTiff -m $MetadataJson `
 $VrtFile -b "" -f

