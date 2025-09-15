import os
from pathlib import Path

import numpy as np
import torch

from terratorch.tasks import SemanticSegmentationTask


def normal_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.output # onnx doesn't support custom model output terratorch.models.model.ModelOutput
    return wrapper


def to_onnx_torch(model, dummy_input, path2onnx):
    import onnx
    import onnxruntime as rt

    def check_onnx(onnx_path):
        model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"{e}")
        else:
            print("passed successsfully")

    print(dummy_input.shape, path2onnx)
    if not os.path.exists(path2onnx):
        torch.onnx.export(
            model,  # Use the underlying PyTorch model
            args=(dummy_input),  # https://github.com/IBM/terratorch/blob/b62fae2edb1f5e3589a5700439a26f7e67c52a01/terratorch/models/backbones/prithvi_mae.py#L418            f=path2onnx,  # Output file name
            export_params=True,  # Store trained parameters
            verbose=False,
            f=path2onnx,
            opset_version=17,  # ONNX opset version
            do_constant_folding=True,  # Optimize constant folding
            input_names=["input"],  # Input tensor name
            output_names=["output"], # Output tensor name
            dynamic_axes={"input": {0: "batch_size"},
                          "output": {0: "batch_size"}})  # add 2: 'width', 3: 'height' ?

   # fix the output of the model is not one !!! type ModelOutput of terratorch (terratorch.models.model.ModelOutput)
    # override forward method without  ModelOutput - https://github.com/IBM/terratorch/blob/b62fae2edb1f5e3589a5700439a26f7e67c52a01/terratorch/models/pixel_wise_model.py#L148
    # torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible.
    # error in this operation https://github.com/IBM/terratorch/blob/b62fae2edb1f5e3589a5700439a26f7e67c52a01/terratorch/models/decoders/upernet_decoder.py#L171 ???
    # similar issue - https://github.com/pytorch/pytorch/issues/74034
    # https://github.com/VCIP-RGBD/DFormer/issues/17  https://github.com/pytorch/pytorch/issues/74034 https://github.com/pytorch/pytorch/issues/42653
    # change function on custom implentation - https://github.com/IBM/terratorch/blob/b62fae2edb1f5e3589a5700439a26f7e67c52a01/terratorch/models/decoders/upernet_decoder.py#L171
    check_onnx(path2onnx)
    """"
    # for prithvi
    import numpy as np 
    class AdaptiveAvgPool2dCustom(nn.Module):
        def __init__(self, output_size):
            super(AdaptiveAvgPool2dCustom, self).__init__()
            self.output_size = np.array(output_size)
    
        def forward(self, x: torch.Tensor):
            stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
            kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
            avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
            x = avg(x)
            return x
    """

    providers = ['CPUExecutionProvider']
    ort_session = rt.InferenceSession(path2onnx, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    print([elm.name for elm in ort_session.get_inputs()]) # ['input']
    print(ort_session.get_inputs()[0]) # NodeArg(name='input', type='tensor(float)', shape=['batch_size', 6, 1, 448, 448])
    onnx_pred = ort_session.run(None, {input_name: dummy_input.numpy(),})
    print(onnx_pred[0].shape)
    results = model(dummy_input)
    print(results.shape)
    results_tf = [results.detach().numpy()]
    for ort_res, tf_res in zip(onnx_pred, results_tf): # Max absolute difference among violations: 0.005127
        np.testing.assert_allclose(ort_res, tf_res, rtol=1e-2, atol=1e-3)


def get_model(config2path, ckpt2path):
    import terratorch
    from terratorch.cli_tools import LightningInferenceModel
    from terratorch.utils import remove_unexpected_prefix
    # def remove_unexpected_prefix(state_dict):
    #     state_dict_ = {}
    #     for k, v in state_dict.items():
    #         keys = k.split(".")
    #         if "_timm_module" in keys:
    #             index = keys.index("_timm_module")
    #             keys.pop(index)
    #             k_ = ".".join(keys)
    #         else:
    #             k_ = k
    #         state_dict_[k_] = v
    #     return state_dict_

    lightning_model = LightningInferenceModel.from_config(config2path)
    """
    Error(s) in loading state_dict for PixelWiseModel:
    	Missing key(s) in state_dict: "neck.2.fpn1.0.weight", "neck.2.fpn1.0.bias", "neck.2.fpn1.1.weight", "neck.2.fpn1.1.bias", "neck.2.fpn1.1.running_mean", "neck.2.fpn1.1.running_var", "neck.2.fpn1.3.weight", "neck.2.fpn1.3.bias", "neck.2.fpn2.0.weight", "neck.2.fpn2.0.bias".
    	Unexpected key(s) in state_dict: "neck.layers.2.fpn1.0.weight", "neck.layers.2.fpn1.0.bias", "neck.layers.2.fpn1.1.weight", "neck.layers.2.fpn1.1.bias", "neck.layers.2.fpn1.1.running_mean", "neck.layers.2.fpn1.1.running_var", "neck.layers.2.fpn1.1.num_batches_tracked", "neck.layers.2.fpn1.3.weight", "neck.layers.2.fpn1.3.bias", "neck.layers.2.fpn2.0.weight", "neck.layers.2.fpn2.0.bias".
    """
    # model = lightning_model.model.load_from_checkpoint(ckpt2path, map_location="cpu")

    weights = torch.load(ckpt2path, map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]

    # It removes a residual prefix (related to timm) from older
    # checkpoints.
    weights = remove_unexpected_prefix(weights)  # "neck.layers.2.fpn1.0.weight"
    # Removing the undesirable prefix `model` from the checkpoint

    # state_dict_ = {}
    # for k, v in weights.items():
    #     keys = k.split(".")
    #     if "neck" in keys and "layers" in keys:
    #         index = keys.index("layers")
    #         keys.pop(index)
    #         k_ = ".".join(keys)
    #     else:
    #         k_ = k
    #     state_dict_[k_] = v
    # weights = state_dict_

    weights_ = {}
    for k, v in weights.items():
        splits = k.split(".")
        if splits[0] == "model":
            splits = splits[1:]
            k_ = ".".join(splits)
            weights_[k_] = v
        else:
            weights_[k] = v

    print(lightning_model.model.model.state_dict().keys() == weights_.keys())

    lightning_model.model.model.load_state_dict(weights_)

    for key_name in weights_.keys():
        np.testing.assert_allclose(lightning_model.model.model.state_dict()[key_name],
                                   weights_[key_name], rtol=1e-5, atol=1e-4)
    return lightning_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert to ONNX model"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="output .onnx file path",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="input checkpoints.ckpt file path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        required=True,
        help="input config.yaml file path",
    )
    parser.add_argument(
        "-s",
        "--input-size",
        type=int,
        required=False,
        default=224 * 4,
        help="input_size of image",
    )
    args = parser.parse_args()

    path2weights: Path = args.input
    path2config: Path = args.config_file
    path2onnx: Path = args.output
    input_size: int = args.input_size

    # lightning_model = LightningInferenceModel.from_config(path2config, path2weights)
    # model = SemanticSegmentationTask.load_from_checkpoint(
    #     path2weights).model # terratorch.tasks.segmentation_tasks.SemanticSegmentationTask
    # PixelWiseModel
    model = get_model(str(path2config), path2weights).model
    model.forward = normal_output(model.forward) # change inside the source code forward output
    model.eval()

    # divisible by the patch
    # size of the 600M models (14 × 14)
    input_shape = (1, 6, input_size, input_size)
    dummy_input = torch.rand(*input_shape).type(torch.float32)  # torch.bfloat16
    to_onnx_torch(model, dummy_input=dummy_input,
                  path2onnx=path2onnx) # lightning_model.model
