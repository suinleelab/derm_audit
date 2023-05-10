#!/usr/bin/env python
"""
Utility to convert ONNX models to PyTorch.
"""
import onnx
import torch
from torch import nn
from onnx2pytorch.convert import extract_attributes

class QuantizeLinear(torch.quantization.FixedQParamsFakeQuantize):
    def __init__(self, node, weights):
        scale_name = node.input[1]
        zero_point_name = node.input[2]
        scale = weights[scale_name].float_data[0]
        zero_point = weights[zero_point_name].int32_data[0]
        super().__init__(scale, zero_point)

    def forward(self, x):
        out = super().forward(x)
        return out

class DequantizeLinear(nn.Identity):
    def forward(self, x):
        out = super().forward(x)
        return out

class Conv(nn.Module):
    def __init__(self, node, weights):
        super().__init__()
        attrs = extract_attributes(node)
        self.pad = attrs['padding']
        if isinstance(self.pad, tuple):
            self.pad = nn.ConstantPad2d(self.pad[0], 0)
        self.stride = attrs['stride']
        self.dilation = attrs['dilation']
        self.groups = attrs['groups']
        self.kernel_size = attrs['kernel_size']

    def forward(self, x, weight, bias):
        temp = self.pad(x)
        return nn.functional.conv2d(temp, weight, bias=bias, stride=self.stride,
                dilation=self.dilation, groups=self.groups)

class Wrapper(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, *inputs):
        return self.f(*inputs)

class ReduceMean(nn.Module):
    def __init__(self, node):
        super().__init__()
        attrs = extract_attributes(node)
        self.dim = attrs['dim']
        self.keepdim = bool(attrs['keepdim'])

    def forward(self, x):
        return torch.mean(x, self.dim, keepdim=self.keepdim)

class Gemm(nn.Module):
    def __init__(self, node, weights):
        super().__init__()
        attrs = extract_attributes(node)
        self.attrs = attrs

    def forward(self, x1, x2, bias):
        if len(x1.shape) > 3 and x1.shape[3] == 1:
            x1 = x1.squeeze(3)
        if len(x1.shape) > 2 and x1.shape[2] == 1:
            x1 = x1.squeeze(2)
        return nn.functional.linear(x1,x2,bias=bias)

class Clip(nn.Module):
    def forward(self, x, min_, max_):
        return torch.clamp(x, min=min_, max=max_)

class AveragePool(nn.Module):
    def __init__(self, node):
        super().__init__()
        self.auto_pad = None
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                self.kernel_shape = tuple(attr.ints)
            if attr.name == 'auto_pad':
                self.auto_pad = attr.s.decode()
            if attr.name == 'strides':
                self.strides = tuple(attr.ints)

    def forward(self, x):
        if self.auto_pad == 'VALID':
            padding = 0
        else:
            raise NotImplementedError
        return nn.functional.avg_pool2d(x, self.kernel_shape, 
                stride=self.strides, padding=padding)

class ConvertedModel(nn.Module):
    '''PyTorch version of onnx model. Only designed for use with Scanoma and 
    Smart Skin Cancer Detection models.'''
    def __init__(self, onnx_model, verbose=False):
        '''Initialize PyTorch model from onnx model.

        Args:
            onnx_model (onnx ModelProto): the model to convert to PyTorch
            verbose(bool, optional): Print information on input and output
                tensors on each forward pass.'''
        super().__init__()
        self.verbose = verbose
        weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}
        self.ops = nn.ModuleList()
        self.in_connections = [] 
        self.out_connections = []
        self.tensors = {}
        self.parameter_dict = nn.ParameterDict()
        graph = onnx_model.graph
        self.input_name = graph.input[0].name
        self.output_name = graph.output[0].name
        for tensor in graph.initializer:
            data = torch.from_numpy(onnx.numpy_helper.to_array(tensor))
            requires_grad = True if data.dtype in [torch.half, torch.float, 
                    torch.double, torch.bfloat16, torch.cfloat, torch.cdouble] else False
            self.parameter_dict[tensor.name.replace('.','')] = nn.Parameter(data, requires_grad=requires_grad)
            self.tensors[tensor.name] = self.parameter_dict[tensor.name.replace('.', '')]
        for node in graph.node:
            if node.op_type == 'QuantizeLinear':
                self.ops.append(QuantizeLinear(node, weights))
                self.in_connections.append([node.input[0]])
                self.out_connections.append(node.output)
            elif node.op_type == 'DequantizeLinear':
                self.ops.append(DequantizeLinear())
                self.in_connections.append([node.input[0]])
                self.out_connections.append(node.output)
            elif node.op_type == 'Conv':
                self.ops.append(Conv(node, weights))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'Add':
                self.ops.append(Wrapper(torch.add))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'ReduceMean':
                self.ops.append(ReduceMean(node))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'Gemm':
                self.ops.append(Gemm(node, weights))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'Softmax':
                self.ops.append(nn.Softmax(dim=1))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'Clip':
                self.ops.append(Clip())
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            elif node.op_type == 'AveragePool':
                self.ops.append(AveragePool(node))
                self.in_connections.append(node.input)
                self.out_connections.append(node.output)
            else:
                raise ValueError("Operation {:s} is not implemented.".format(str(node.op_type)))

    def forward(self, x):
        self.tensors[self.input_name] = x
        for i_op, op in enumerate(self.ops): 
            input_names = self.in_connections[i_op]
            output_names = self.out_connections[i_op]
            if self.verbose:
                print("op: ", op.__class__)
                print("  inputs:")
                for name in input_names:
                    print("    ", name, ":", self.tensors[name].shape)
                print("  outputs: ", output_names)
            in_tensors = [self.tensors[name] for name in input_names]
            result = self.ops[i_op](*in_tensors)
            if isinstance(result, list):
                for idx in range(len(output_names)):
                    name = output_names[idx]
                    self.tensors[name] = result[idx]
            else:
                self.tensors[output_names[0]] = result
        return self.tensors[self.output_name]
