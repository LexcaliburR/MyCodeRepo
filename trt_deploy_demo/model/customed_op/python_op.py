'''
Author: lexcalibur
Date: 2021-12-14 14:51:59
LastEditors: lexcaliburr
LastEditTime: 2021-12-14 14:51:59
'''

import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic

class MLineImplementation(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input, weight):
        return g.op("my_ops::mline", input, weight, description_s="this is customed linear op")

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input)
        # can call some cuda function
        return input * weight

class MLine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(out_channels, in_channels))
        self.line = MLineImplementation()

    def forward(self, x):
        return self.line.apply(x, self.weight)


def testMLine():
    line = MLine(3, 3)
    input = torch.randn([3, 3])
    print(input)
    print(line(input))


def exportONNX():
    input = torch.randn([3, 3])
    model = MLine(3, 3)
    torch.onnx.export(model, input, "mline_pythonop.onnx", verbose=True, opset_version=9)

exportONNX()

