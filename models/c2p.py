#!/usr/bin/env python
"""
Utility to convert Caffe model to PyTorch.
"""
import google.protobuf.descriptor
import google.protobuf.text_format
import torch
from torch import nn

import models.caffe_pb2 as caffe_pb2
import models.modelderm_labels as modelderm_labels

class Pool(nn.Module):
    '''Caffe-compatible pooling layer. 

    In contrast to PyTorch, which drops half windows, Caffe bottom-pads and/or 
    right-pads the input to fill partial windows.

    Args:
        pooling_type (int): 0 for max pooling, or 1 for average pooling. 
            Integers match caffe naming scheme.
        kernel_size (int): The size of the pooling kernel (width and height 
            will be equal).
        stride: (int): The stride of the pooling operation.'''
    def __init__(self, pooling_type, kernel_size, stride):
        super().__init__()
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        if self.pooling_type == 0:
            layer = nn.MaxPool2d
        elif self.pooling_type == 1:
            layer = nn.AvgPool2d
        else:
            raise NotImplementedError("Pooling type {:d} not implemented"
                                      .format(pooling_type))
        self.pool = layer(kernel_size, stride=stride)
        # initialize padding; only use in forward pass if necessary
        self.pad = nn.ZeroPad2d((0,1,0,1))  

    def forward(self, x):
        if self.stride == 2:
            x = self.pad(x)
        return self.pool(x)

class Scale(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            bias = None
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x*self.weight.unsqueeze(1).unsqueeze(2) + self.bias.unsqueeze(1).unsqueeze(2)

class EltwiseSum(nn.Module):
    def forward(self, x, y):
        return x + y

class InnerProduct(nn.Module):
    def forward(self, x):
        if x.shape[2] == 1 and x.shape[3] == 1:
            x = x.squeeze(3).squeeze(2)
        return torch.nn.functional.linear(x, self.weight) + self.bias

class ConvertedCaffeModel(nn.Module):
    """PyTorch implementation of caffe model.
    
    Args:
        proto_path (str): The file path to the `.proto` file that defines the 
            caffe model.
        weights_path (str): The file path to the caffe model weights file (in 
            caffemodel format)."""
    def __init__(self, proto_path, weights_path):
        super().__init__()

        self.labels = modelderm_labels.LABELS 
        self.net = caffe_pb2.NetParameter()
        self.tensors = {}
        self.ops = nn.ModuleDict()
        self.in_connections = []
        self.out_connections = []

        # Parse network and initialize PyTorch version of each layer
        with open(proto_path, 'r') as f:
            google.protobuf.text_format.Parse(f.read(), self.net)
        for layer in self.net.layer:
            if layer.type == 'Input':
                self.in_connections.append(['input'])
                self.ops[layer.name] = nn.Identity()
                self.out_connections.append([layer.top[0]])
                self.input_dim = layer.input_param.shape[0].dim[1]
            elif layer.type == 'Convolution':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = nn.Conv2d(self.input_dim, 
                                          layer.convolution_param.num_output,
                                          layer.convolution_param.kernel_size,
                                          stride=layer.convolution_param.stride,
                                          padding=layer.convolution_param.pad,
                                          bias=False)
            elif layer.type == 'BatchNorm':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                for module in self.ops: # get last item
                    pass
                n_channels = self.ops[module].out_channels
                self.ops[layer.name] = nn.BatchNorm2d(n_channels, affine=False)
            elif layer.type == 'Scale':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = Scale(bias=layer.scale_param.bias_term)
            elif layer.type == 'ReLU':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = nn.ReLU()
            elif layer.type == 'Pooling':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = Pool(layer.pooling_param.pool,
                                            layer.pooling_param.kernel_size,
                                            layer.pooling_param.stride)
            elif layer.type == 'Eltwise': # assume elementwise sum
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = EltwiseSum()
            elif layer.type == 'InnerProduct':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = InnerProduct()
            elif layer.type == 'Softmax':
                self.in_connections.append(layer.bottom)
                self.out_connections.append(layer.top)
                self.ops[layer.name] = nn.Softmax(dim=1)
            else:
                raise NotImplementedError('Layer type {:s} not implemented'.format(layer.type))

        # Load weights into "blobs"
        with open(weights_path, 'rb') as f:
            parsed_weights = self.net.ParseFromString(f.read())

        # Iterate through layers and set weights
        for i, layer in enumerate(self.net.layer):
            try:
                op = self.ops[layer.name]
                for iblob, blob in enumerate(layer.blobs):
                    if len(blob.shape.dim) > 0:
                        shape = list(blob.shape.dim)
                    else:
                        shape = [blob.num, blob.channels, blob.height, blob.width]
                    tensor = nn.Parameter(torch.tensor(blob.data).view(shape))
                    if layer.type == 'Convolution':
                        if iblob == 0:
                            op.weight = tensor 
                        if iblob == 1:
                            raise NotImplementedError("Bias for convolution layer not implemented")
                    elif layer.type == 'BatchNorm':
                        if iblob == 0:
                            op.running_mean = torch.tensor(tensor)
                        elif iblob == 1:
                            op.running_var = torch.tensor(tensor)
                        elif iblob == 2 and not all(tensor == 1):
                            raise ValueError("Handling of third parameter for batchnorm with value {:s} undefined.".format(tensor))
                    elif layer.type == 'Scale':
                        if iblob == 0:
                            op.weight = tensor
                        elif iblob == 1:
                            op.bias = tensor
                        else:
                            raise ValueError("Handling of third parameter for scale undefined.")
                    elif layer.type == 'InnerProduct':
                        if iblob == 0:
                            op.weight = tensor
                        elif iblob == 1:
                            op.bias = tensor
                    else:
                        raise NotImplementedError("Parameter setting for layer {:s} not implemented".format(layer.type))
            except KeyError:
                pass

    def forward(self, x, verbose=False):
        self.tensors['input'] = x
        for i_op, op_name in enumerate(self.ops):
            op = self.ops[op_name]
            in_names = self.in_connections[i_op]
            out_names = self.out_connections[i_op]
            if verbose: print("Op {:s} from {:s} to {:s}".format(op_name, str(in_names), str(out_names)))
            ins = [self.tensors[name] for name in in_names]
            if verbose: 
                print("  inputs:")
                for name, tensor in zip(in_names, ins):
                    print("    ", name, ": ", tensor.shape)
            outs = op(*ins)
            if verbose: print("  outputs:")
            if len(out_names) > 1:
                for i, name in enumerate(out_names):
                    self.tensors[name] = outs[i]
                    if verbose: print("    ", name, ": ", outs[i].shape)
            else:
                self.tensors[out_names[0]] = outs
                if verbose: print("    ", out_names[0], ": ", outs.shape)
        return outs

def test():
    model = ConvertedCaffeModel('./deploy.prototxt', '70616.caffemodel')
    dummy_input = torch.rand((4,3,224,224))
    positive_index = model.labels.index("malignantmelanoma")
    p = model(dummy_input)
    print("p:", p)
    print("mela: ", p[:, positive_index])

if __name__ == "__main__":
    test()
