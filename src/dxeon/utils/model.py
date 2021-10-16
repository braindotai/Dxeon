from typing import Union
from dataclasses import dataclass, InitVar, field
import numpy as np
import torch
from torch import nn
import onnxruntime
from time import time

def freeze(model: nn.Module) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False

def unfreeze(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_()

def load_model(model: nn.Module, state_dict_path: str, strict: bool = True) -> None:
    model.load_state_dict(torch.load(state_dict_path, map_location = 'cpu'), strict = strict)

def save_model(model: nn.Module, state_dict_path: str) -> None:
    torch.save(model.state_dict(), state_dict_path)

@torch.no_grad()
def benchmark_performance(model: nn.Module, input_batch: torch.Tensor, runs = 30, cudnn_benchmark: bool = False) -> None:
    torch.backends.cudnn.benchmark = cudnn_benchmark

    model(input_batch)
    model(input_batch)

    tmps = []
    for _ in range(runs):
        start = time()
        model(input_batch)
        took = time() - start
        tmps.append(took)
    
    tmps = np.array(tmps)

    print('[=================== Batch Stats ================== ]')
    print(f'Avg seconds per batch     \t: {tmps.mean():.3f}')
    print(f'Median seconds per batch  \t: {np.median(tmps):.3f}')
    
    print(f'Avg Batches per second    \t: {(1 / tmps.mean()):.3f}')
    print(f'Median Batches per second \t: {(1 / np.median(tmps)):.3f}')
    
    print('\n[================== Sample Stats ================== ]')
    print(f'Avg seconds per sample    \t: {(tmps.mean() / input_batch.size(0)):.3f}')
    print(f'Median seconds per sample \t: {(np.median(tmps) / input_batch.size(0)):.3f}')
    
    print(f'Avg Samples per second    \t: {(input_batch.size(0) / tmps.mean()):.3f}')
    print(f'Median Samples per second \t: {(input_batch.size(0) / np.median(tmps)):.3f}')

@dataclass
class ONNXModel:
    onnx_path: InitVar[str]
    session: onnxruntime.InferenceSession = field(init = False)

    def __post_init__(self, onnx_path: str) -> None:
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(onnx_path, sess_options = options)
        # self.session.set_providers(['CUDAExecutionProvider'])

        # self.session = onnxruntime.InferenceSession(onnx_path)
    
    def __call__(self, inputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.session.run(
            None,
            {self.session.get_inputs()[0].name: inputs if isinstance(inputs, np.ndarray) else inputs.numpy()}
        )[0]


@dataclass
class GenerateONNXModel:
    nn_model: InitVar[nn.Module]
    state_dict_path: InitVar[str]
    onnx_path: str
    input_shape: tuple
    opset_version: int = 9
    session: onnxruntime.InferenceSession = field(init = False)

    def __post_init__(self, nn_model: nn.Module, state_dict_path: str = None) -> None:
        x = torch.ones(*self.input_shape)
        nn_model.eval()
        if state_dict_path:
            load_model(nn_model, state_dict_path, False)
        torch.onnx.export(
            nn_model,
            x,
            self.onnx_path,
            export_params = True,
            opset_version = self.opset_version,
            do_constant_folding = True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {
                'input': {0 : 'batch_size'},
                'output': {0 : 'batch_size'}
            },
        )

        self.session = onnxruntime.InferenceSession(self.onnx_path)
    
    def __call__(self, inputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.session.run(
            None,
            {self.session.get_inputs()[0].name: inputs if isinstance(inputs, np.ndarray) else inputs.numpy()}
        )[0]