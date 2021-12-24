'''
Author: lexcalibur
Date: 2021-12-06 15:06:15
LastEditors: lexcaliburr
LastEditTime: 2021-12-07 09:37:39
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyExpFunc(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input):
        return g.op('MyExp', input)

    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


class MyExp(nn.Module):
    def forward(self, x):
        return MyExpFunc.apply(x)


class Vergross(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.line1 = nn.Linear(10, 10)
        # self.line2 = nn.Linear(10, 10)
        self.exp = MyExp()

    def forward(self, x):
        new_data = torch.ones(10) * 2
        x = self.exp(x)
        # x = F.normalize(x, dim=0)
        x = torch.cat([x.reshape([-1, 10]), new_data.reshape([-1, 10])])
        return x


class MyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.line1 = nn.Linear(10, 10)
        self.vergross = Vergross()

    def forward(self, x):
        x = self.line1(x)
        x = self.vergross(x)
        return x


if __name__ == '__main__':
    model = MyNet()
    print(model(torch.ones(10)))
    torch.onnx.export(model, (torch.ones(10)), "./mynet.onnx_customed_layer",
                      input_names=['input'],
                      output_names=['output'],
                      verbose=True,
                      )

# def export(model, args, f, export_params=True, verbose=False, training=False,
#            input_names=None, output_names=None, aten=False, export_raw_ir=False,
#            operator_export_type=None, opset_version=None, _retain_param_name=True,
#            do_constant_folding=True, example_outputs=None, strip_doc_string=True,
#            dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
#            enable_onnx_checker=True, use_external_data_format=False):
