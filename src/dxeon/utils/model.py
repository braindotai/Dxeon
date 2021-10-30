from typing import Union
from dataclasses import dataclass, InitVar, field
import numpy as np
import torch
from torch import nn
import torch.nn.utils.prune as prune
import onnxruntime
from time import time
from tqdm.auto import tqdm

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

def get_device(model: nn.Module):
    return next(model.parameters()).device

@torch.no_grad()
def benchmark_performance(model: nn.Module, input_batch: torch.Tensor, runs = 20, cudnn_benchmark: bool = False) -> None:
    torch.backends.cudnn.benchmark = cudnn_benchmark

    model(input_batch)
    model(input_batch)

    tmps = []
    loader = tqdm(range(runs), desc = 'Benchmarking Completed', ncols = 200)
    for _ in loader:
        start = time()
        model(input_batch)
        took = time() - start
        loader.set_postfix(latest_runtime = f'{took:.3f} secs')
        tmps.append(took)
    
    tmps = np.array(tmps)

    print(f'\n[============ Batch({input_batch.size(0)} samples) Stats ============]')
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
class LoadONNXModel:
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
        nn_model.eval().cpu()
        
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


def prune_l1_unstructured(model, layer_types, param_types, proportion):
    for module in model.modules():
        if any([isinstance(module, layer_type) for layer_type in layer_types]):
            for param_type in param_types:
                if getattr(module, param_type) is not None:
                    prune.l1_unstructured(module, param_type, proportion)
                    prune.remove(module, param_type)

    return model

def prune_l1_structured(model, layer_types, param_types, proportion):
    for module in model.modules():
        if any([isinstance(module, layer_type) for layer_type in layer_types]):
            for param_type in param_types:
                if getattr(module, param_type) is not None:
                    prune.ln_structured(module, param_type, proportion, n = 1, dim = 1)
                    prune.remove(module, param_type)

    return model

def prune_global_unstructured(model, layer_types, param_types, proportion):
    module_tups = []
    for module in model.modules():
        if any([isinstance(module, layer_type) for layer_type in layer_types]):
            for param_type in param_types:
                if getattr(module, param_type) is not None:
                    # print(module, param_type)
                    module_tups.append((module, param_type))
    
    prune.global_unstructured(
        parameters = module_tups,
        pruning_method = prune.L1Unstructured,
        amount = proportion
    )

    for module, param_type in module_tups:
        prune.remove(module, param_type)

    return model

def compute_sparsity(model: nn.Module, layer_types, param_types):
    total_params_sum = 0
    total_params_count = 0
    
    for module in model.modules():
        if any([isinstance(module, layer_type) for layer_type in layer_types]):
            for param_type in param_types:
                if getattr(module, param_type) is not None:
                    total_params_sum += torch.sum(module.weight == 0)
                    total_params_count += module.weight.nelement()
    
    return 100.0 * float(total_params_sum / float(total_params_count))

    '''
    layers_to_prune = [nn.Conv2d]
    params_to_prune = ['weight', 'bias']

    print(f'Model sparsity before pruning\t: {compute_sparsity(model, layers_to_prune, params_to_prune):.3}%')
    '''