'''
Author: lexcalibur
Date: 2021-12-13 15:52:38
LastEditors: lexcaliburr
LastEditTime: 2021-12-13 17:30:43
'''
import torch
from torch.onnx.symbolic_helper import parse_args


torch.ops.load_library("/home/lishiqi/myrepos/MyCodeRepo/trt_deploy_demo/model/PluginDemo/torch_op/build/libhswish.so")

print(torch.ops.my_ops.hswish)
print(torch.ops.my_ops.hswish(torch.zeros(3), torch.ones(3), torch.ones(3)))

print(" ------------ relu -----------------")
input1 = torch.randn(3)
print("input1= ", input1)
print(torch.ops.my_ops.mrelu)
output = torch.ops.my_ops.mrelu(input1, torch.ones(3))
print("mrelu output= ", output)

print("---------- trace ------------------------------")

# def comput(g, x1, x2, x3, y1, y2):
#     # x = torch.ops.my_ops.hswish(x1, x2, x3) + torch.ops.my_ops.mrelu(y1, y2)
#     x = g.op("my_ops::hswish", x1, x2, x3) + g.op("my_ops::mrelu", y1, y2)
#     return x
@parse_args('v', 't', 't')
def hswish(g, x1, x2, x3):
    return g.op("my_ops::hswish", x1, x2, x3)

@parse_args('v', 't')
def mrelu(g, x1, x2):
    return g.op("my_ops::mrelu", x1, x2)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('my_ops::hswish', hswish, 9)
register_custom_op_symbolic('my_ops::mrelu', mrelu, 9)

class DemoCom(torch.nn.Module):
    def __init__(self):
        super(DemoCom, self).__init__()
        # self.com = comput
        self.weight1 = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.ops.my_ops.hswish(x[0], self.weight1, self.weight1) + torch.ops.my_ops.mrelu(x[3], self.weight1)


inputs = [torch.ones(3), torch.ones(3), torch.ones(3), torch.ones(3), torch.ones(3)]
model = DemoCom()
torch.onnx.export(model, inputs, "DemoCom.onnx", export_params=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, verbose=True)

