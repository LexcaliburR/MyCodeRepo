import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
from pycuda.compiler import SourceModule

# CUDA 核函数，增加复杂度
mod = SourceModule("""
__global__ void complex_compute(float *dest, float *a, float *b, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float temp = 0;
    for (int j = 0; j < 100; ++j) {  // 增加循环以增加计算复杂度
        temp += a[i] * b[i];
        temp = temp * 1.0001f + 0.0001f;  // 模拟复杂的计算过程
    }
    dest[i] = temp;
  }
}
""")

complex_compute = mod.get_function("complex_compute")

# 初始化数据
n_elements = 16000000  # 16M 元素
a = np.random.randn(n_elements).astype(np.float32)
b = np.random.randn(n_elements).astype(np.float32)
dest = np.zeros_like(a)

block_size = 256  # 每个线程块的线程数
grid_size = (len(a) + block_size - 1) // block_size  # 计算网格大小

# CUDA 计算时间测量
start_cuda = time.time()
complex_compute(
    drv.Out(dest), drv.In(a), drv.In(b), np.int32(len(a)),
    block=(block_size, 1, 1), grid=(grid_size, 1)
)
drv.Context.synchronize()  # 确保内核执行完成
end_cuda = time.time()

# NumPy 计算时间测量（增加相同的复杂计算）
start_numpy = time.time()
numpy_result = np.zeros_like(a)
for i in range(100):  # 模拟相同的复杂度
    numpy_result += a * b
    numpy_result = numpy_result * 1.0001 + 0.0001
end_numpy = time.time()

# 打印结果和时间
print("CUDA 计算时间: {:.6f} 秒".format(end_cuda - start_cuda))
print("NumPy 计算时间: {:.6f} 秒".format(end_numpy - start_numpy))

# 验证结果一致性
if np.allclose(dest, numpy_result, atol=1e-3):
    print("结果一致！")
else:
    print("结果不一致！")