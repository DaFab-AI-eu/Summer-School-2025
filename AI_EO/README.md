# Workflow for DaFab feature extraction

## How to?

### Run the workflow

- `workflow_ai_water_detect.ps1` to fetch metadata (+ assets) from AWS and compute water detection

For Prithvi/TerraMind model (`-SubName prithvi` \ `-SubName terramind`) and `-Model` checkpoints (https://drive.google.com/drive/folders/1Z5UgCCamR5O8RnkhQhLLdOraOY1ZyF5o) and flag `-DebugVisualization` to visualize RGB image and Scene Classification Map
```
.\workflow_ai_water_detect.ps1 -Model "..\..\checkpoints\prithvi_v2\Prithvi-EO-V2-300M-TL-Sen1Floods11_s_896_without_clouds.onnx" -CacheFolder "..\..\val" -TemporaryFolder "..\..\val" -SubName prithvi -PatchCount -1 -PatchOverlap 150
```

## Some tile ID to test the workflow on

- S2A_10SGG_20230426_0_L2A
- S2B_38NNL_20231120_0_L2A
- S2A_46QHH_20240916_0_L2A
- S2B_46QHE_20231004_0_L2A
- S2A_48PWC_20201022_0_L2A
- S2A_48QTE_20240917_0_L2A
- S2B_45RYJ_20200802_0_L2A
- S2A_31PBS_20230923_0_L2A
- S2B_45RYJ_20200802_0_L2A
- S2B_30SYJ_20241110_0_L2A
- S2C_30SYJ_20250213_0_L2A

## Conda Environment for inference workflow

Added an `environment.yml` file which can be used to easily build (`conda env create -f environment.yml`) a conda environment named *dafab* to ensure most of the necessary modules are present for the workflow (some still need to be installed manually).
