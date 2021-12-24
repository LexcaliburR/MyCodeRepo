'''
Author: lexcalibur
Date: 2021-12-09 17:41:26
LastEditors: lexcaliburr
LastEditTime: 2021-12-13 17:03:21
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import json

# torch.ops.load_library("/home/lishiqi/myrepos/MyCodeRepo/trt_deploy_demo/model/PluginDemo/torch_op/build/libhswish.so")

# 一个内置插件
class HSwishImplementation(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, weight, bias):
        b = torch.ones(1)
        return g.op("HSwish", input, weight, bias, b, info_s="string attribute", test_property_s="this is test", kernel_size_i=3, eps_f=3e-2)

    @staticmethod
    def forward(ctx, i, weight, bias):
        ctx.save_for_backward(i)
        # can call some cuda function
        return i * F.relu6(i + 3) / 6 + weight + bias

class MemoryEfficientHSwish(nn.Module):
    def __init__(self, shape):
        super(MemoryEfficientHSwish, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))
        self.bias.data.fill_(3.15)
        self.weight = nn.Parameter(torch.zeros(shape))
        self.weight.data.fill_(99999)

    def forward(self, x):
        return HSwishImplementation.apply(x, self.weight, self.bias)


# 一个通过本框架实现的插件
class MReLUImplementation(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, bias):
        return g.op("Plugin", input, bias, name_s="MReLU", info_s=json.dumps({
            "kernel_size": 3,
            "eps": 3e-2,
            "other": "Hello Onnx Plugin"
        }))

    @staticmethod
    def forward(ctx, i, bias):
        ctx.save_for_backward(i)
        return F.relu(i) + bias

class MReLU(nn.Module):
    def __init__(self, *shape):
        super(MReLU, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))
        self.bias.data.fill_(0.5)

    def forward(self, x):
        return MReLUImplementation.apply(x, self.bias)

class FooModel(torch.nn.Module):
    def __init__(self):
        super(FooModel, self).__init__()
        self.hswish = MemoryEfficientHSwish(1)
        self.mrelu = MReLU(1)

    def forward(self, input1, input2):
        return self.mrelu(input2) + self.hswish(input1)

dummy_input1 = torch.zeros((1, 3, 3, 3))
dummy_input2 = torch.zeros((1, 3, 3, 3))
model = FooModel()

dummy_input1[...] = 0.25
dummy_input2[...] = 0

# pt 1.10
# ONNX
# ONNX_ATEN
# ONNX_ATEN_FALLBACK
# ONNX_FALLTHROUGH

# pt 1.5.1
# ONNX
# ONNX_ATEN
# ONNX_ATEN_FALLBACK
# RAW

torch.onnx.export(model, (dummy_input1, dummy_input2),
                  '/home/lishiqi/myrepos/MyCodeRepo/trt_deploy_demo/model/PluginDemo/onnx/demo.onnx',
                  verbose=True,
                  enable_onnx_checker=True,
                  export_params=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)