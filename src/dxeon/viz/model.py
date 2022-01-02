import os
from typing import List, Tuple, Union
import torch
from torch import nn
import onnx
from onnx.tools.net_drawer import BLOB_STYLE, GetPydotGraph, GetOpNodeProducer

from PIL import Image

BLOB_STYLE['shape'] = 'box3d'

OP_STYLE = {
    'shape': 'box3d',
    'color': 'black',
    'fontcolor': 'black',
    'margin': 0.3
}


@torch.no_grad()
def model(model: nn.Module, input_size: Union[List[int], Tuple[int]], remove_png = True):
    onnx_path = "temp_model.onnx"
    torch.onnx.export(model, torch.ones(*input_size).to(next(model.parameters()).device), onnx_path)
    onnx_model = onnx.load(onnx_path)
    os.remove(onnx_path)
    pydot_graph = GetPydotGraph(
        onnx_model.graph,
        name = onnx_model.graph.name,
        rankdir = "TB",
        node_producer = GetOpNodeProducer("docstring", **OP_STYLE)
    )
    pydot_graph.write_dot("graph.dot")
    os.system('dot -O -Tpng graph.dot')
    os.remove('graph.dot')
    img = Image.open('graph.dot.png')
    img.show()
    if remove_png:
        os.remove('graph.dot.png')
    return img