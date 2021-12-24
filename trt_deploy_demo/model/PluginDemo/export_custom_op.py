'''
Author: lexcalibur
Date: 2021-12-13 17:30:01
LastEditors: lexcaliburr
LastEditTime: 2021-12-13 17:31:58
'''

# Create custom symbolic function
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic
import torch

torch.ops.load_library("/home/lishiqi/myrepos/MyCodeRepo/trt_deploy_demo/model/PluginDemo/torch_op/build/libhswish.so")

@parse_args('v', 'v', 't', 't')
def symbolic_hswish(g, input1, attr1, attr2):
    return g.op("hswish", input1, attr1_f=attr1, attr2_f=attr2)

# Register custom symbolic function
# register_custom_op_symbolic('my_ops::hswish', symbolic_hswish, 9)

# register_custom_op_symbolic('custom_ops::foo_forward', symbolic_foo_forward, 9)

class FooModel(torch.nn.Module):
    def __init__(self, attr1, attr2):
        super().__init__()
        self.attr1 = attr1
        self.attr2 = attr2

    def forward(self, input1):
        # Calling custom op
        return torch.ops.my_ops.hswish(input1, self.attr1, self.attr2)

# model = FooModel(torch.ones(3), torch.ones(3))
# torch.onnx.export(model, torch.randn(3), 'model.onnx', custom_opsets={"custom_domain": 2}, verbose=True)
# a = torch.ops
# print(torch.ops)
# print(torch.my_ops)

print(1)