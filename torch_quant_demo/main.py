import torch
from mmdet3d.evaluation.functional.scannet_utils.util_3d import Instance
from torchsummary import summary
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
from torch.nn.intrinsic import ConvReLU2d
from torch.nn import Conv2d, ReLU, BatchNorm2d
from torch.ao.quantization import MinMaxObserver

def export_to_onnx(model):
    pass

def print_model_parameters(model, prefix=""):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if list(module.children()):  # 如果还有子模块
            print_model_parameters(module, full_name)
        else:  # 叶子模块
            print(f"\nModule: {full_name} ({type(module).__name__})")
            for param_name, param in module.named_parameters(recurse=False):
                print(f"  {param_name}: shape={tuple(param.shape)}")
                if param.numel() <= 100:  # 如果参数较少，打印具体值
                    print(f"    values: {param.detach().cpu().numpy().round(4)}")
def print_head(model_name):
    header = f"\n\n\033[1;33m{'-'*88}\033[0m"
    title = f"\033[1;33m| {' '*( (87 - len(model_name)) // 2 )}{model_name}{' '*( (87 - len(model_name)) // 2 )} |\033[0m"
    footer = f"\033[1;33m{'-'*88}\033[0m"
    print(header)
    print(title)
    print(footer)

def visualModel(model, input_tensor, model_name):
    header = f"\n\n\033[1;33m{'-'*88}\033[0m"
    title = f"\033[1;33m| {' '*( (87 - len(model_name)) // 2 )}{model_name}{' '*( (87 - len(model_name)) // 2 )} |\033[0m"
    footer = f"\033[1;33m{'-'*88}\033[0m"
    print(header)
    print(title)
    print(footer)

    # 1. summary
    summary(model, input_size=input_tensor.shape[1:])  # 假设输入为 3通道 224x224 图像

    # 2. tensorboard
    writer = SummaryWriter("logs")
    writer.add_graph(model, input_tensor)
    writer.close()

    # 3. torchviz, torchview
    model_graph = draw_graph(
        model,
        input_size=input_tensor.shape,
        device=next(model.parameters()).device,
    )
    model_graph.visual_graph.render(model_name, format="png")

    print_model_parameters(model)

class MQuant(torch.nn.Module):
    def __init__(self):
        super(MQuant, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn1.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.bn1.bias = torch.nn.Parameter(torch.tensor(0.1))
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.bn2.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.bn2.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.relu2 = torch.nn.ReLU()


    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dequant(x)
        return x

# 1. build f32 model
dev = "cuda" if torch.cuda.is_available() else "cpu"
model_f32 = MQuant().to(dev)
model_f32.eval()
x = torch.randn(1, 3, 224, 224).to(dev)  # 模拟输入
visualModel(model_f32, x, "model_f32")
torch.save(model_f32.state_dict(), "model_f32.pth")

# 2. set quantization config
# 注意这里dtype和具体的硬件相关，例如地平线的NPU使用int8，FPGA使用uint8
custom_qconfig = torch.ao.quantization.QConfig(
    activation=MinMaxObserver.with_args(
        qscheme=torch.per_tensor_affine,
        dtype=torch.quint8,
        quant_min=0,
        quant_max=200,
        reduce_range=False  # 对低精度设备更友好
    ),
    weight=MinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric,
        dtype=torch.qint8,
        quant_min=-100,
        quant_max=100,
        reduce_range=False
    )
)
model_f32.qconfig = custom_qconfig

# 3. fuse modules
# model_f32_fused = torch.quantization.fuse_modules(model_f32, [['conv', 'bn', 'relu']])
model_f32_fused = torch.quantization.fuse_modules(model_f32, [['conv1', 'bn1'], ['conv2', 'bn2']])

visualModel(model_f32_fused, x, "model_f32_fused")

# 4. prepare model
model_f32_prepared = torch.quantization.prepare(model_f32_fused, inplace=False)
visualModel(model_f32_prepared, x, "model_f32_prepared")

print(model_f32_prepared)


# 5. calibration
calibration_data = [torch.randn(1, 3, 224, 224).cuda() for _ in range(10)]  # 模拟数据
model_f32_prepared.eval().cuda()
with torch.no_grad():
    for data in calibration_data:
        model_f32_prepared(data)

print_head("model_f32_prepared after calibration")
print(model_f32_prepared)

# 6. convert to quantized model
model_int8 = torch.ao.quantization.convert(model_f32_prepared)
torch.save(model_int8.state_dict(), "model_int8.pth")
