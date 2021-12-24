'''
Author: lexcalibur
Date: 2021-12-14 15:37:06
LastEditors: lexcaliburr
LastEditTime: 2021-12-14 16:15:16
'''
import torch
import torch.nn as nn
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

torch.ops.load_library("/home/lishiqi/myrepos/MyCodeRepo/trt_deploy_demo/model/customed_op/build/libmline.so")


#################################################################
################# CALL CPP OPS
#################################################################

class SimpleClassifier(torch.nn.Module):
    def __init__(self, in_channel, num_class):
        super().__init__()
        self.line = nn.Linear(in_channel, 16)
        self.mline_weight = torch.nn.Parameter(torch.ones(16, num_class))

    def forward(self, x):
        assert len(x.shape) == 2
        # Calling custom op
        x = self.line(x)
        return torch.ops.my_ops.mline(x, self.mline_weight)

def test_call_cpp_op():
    model = SimpleClassifier(3, 10)
    input = torch.randn([5, 3])
    output = model(input)
    print(output.shape)

test_call_cpp_op()

# ai ding yi mline zhiqian daochu bao cuo
#################################################################
###### EXPORT
#################################################################

@parse_args("v", "v")
def mline(g, input, weight):
    return g.op("my_ops::mline", input, weight)

register_custom_op_symbolic("my_ops::mline", mline, 9)

def exportONNX():
    model = SimpleClassifier(3, 10)
    input = torch.randn([5, 3])
    torch.onnx.export(model, input, "mline_cpp.onnx")

exportONNX()