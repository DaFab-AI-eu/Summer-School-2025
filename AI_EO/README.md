# Workflow for DaFab feature extraction

## How to?

### Run the workflow

`workflow_ai_water_detect.ps1` to fetch metadata (+ assets) from AWS and compute water detection

For Prithvi/TerraMind model (`-SubName prithvi` \ `-SubName terramind`) and `-Model` checkpoints (https://drive.google.com/drive/folders/1Z5UgCCamR5O8RnkhQhLLdOraOY1ZyF5o) and flag `-DebugVisualization` to visualize RGB image and Scene Classification Map

Windows
```
.\workflow_ai_water_detect.ps1 -Model "..\..\checkpoints\prithvi_v2\Prithvi-EO-V2-300M-TL-Sen1Floods11_s_896_without_clouds.onnx" -CacheFolder "..\..\val" -TemporaryFolder "..\..\val" -SubName prithvi -PatchCount -1 -PatchOverlap 150
```
Linux

```
path2dir=..
chmod +x "./Summer-School-2025/AI_EO/workflow_ai_water_detect.sh"
cd ./Summer-School-2025/AI_EO && ./workflow_ai_water_detect.sh S2A_46QHH_20240916_0_L2A \ 
$path2dir/cache $path2dir/temp $path2dir/terramind_sen1floods11.onnx terramind -1 2
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

## ONNX convert of terratorch model
` conda create -n py3_12 python=3.12` 

 `conda activate py3_12`

`pip install git+https://github.com/IBM/terratorch.git`

`pip install onnx onnxruntime` 

Run the script

`python .\convert2onnx_terratorch.py -i "terramind_v1\\checkpoints.ckpt" -c "terramind_v1\\config.yaml" -o "terramind_v1_base_sen1floods11_s_896_without_clouds.onnx"` 

# Run Geo Foundation Model finetuning & inference

notebook `Water_detection_v1.ipynd`


1. Setup [5 min]
  *   Runtime
  *   Installations
  *   Mount Google Drive
  *   Dataset download

2. Sen1Floods11 Dataset [15 min]

  *   Overview
  *   Visualizations
  *   Dataset preparation
3. Finetuning Geo Foundation Models (TerraMind / Prithvi) [20 min]

  *   Tensorboard
  *   Configuration file
  *   Finetuning for 5 epochs
  *   Test metrics for trained model

4. Inference

Sentinel-2 product search [10 min]
- Web interface
- STAC API in Python

Inference workflow on Sentinel-2 product [20 min]
- ONNX conversion
- Inference steps explanation
- Run
- Visualizations outputs

5. Q & A [10 min]