import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import time
from pycuda.compiler import SourceModule

# 输入图像尺寸
INPUT_WIDTH = 7680
INPUT_HEIGHT = 4320
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 512

# 普通全局内存的 Resize 核函数
resize_global_kernel = """
__global__ void resize_global(const float *input, float *output, int in_width, int in_height, int out_width, int out_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        float src_x = x * (float)in_width / out_width;
        float src_y = y * (float)in_height / out_height;
        int src_x0 = (int)src_x;
        int src_y0 = (int)src_y;

        // 最近邻插值
        output[y * out_width + x] = input[src_y0 * in_width + src_x0];
    }
}
"""

# 使用纹理内存的 Resize 核函数
resize_texture_kernel = """
texture<float, 2, cudaReadModeElementType> tex;

__global__ void resize_texture(float *output, int out_width, int out_height, float scale_x, float scale_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        float src_x = x * scale_x;
        float src_y = y * scale_y;

        // 从纹理中读取数据
        output[y * out_width + x] = tex2D(tex, src_x, src_y);
    }
}
"""

# 编译核函数
mod_global = SourceModule(resize_global_kernel)
resize_global = mod_global.get_function("resize_global")

mod_texture = SourceModule(resize_texture_kernel)
resize_texture = mod_texture.get_function("resize_texture")

# 初始化输入图像
input_image = np.random.rand(INPUT_HEIGHT, INPUT_WIDTH).astype(np.float32)
output_image_global = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)
output_image_texture = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH), dtype=np.float32)

# 计算缩放比例
scale_x = float(INPUT_WIDTH) / OUTPUT_WIDTH
scale_y = float(INPUT_HEIGHT) / OUTPUT_HEIGHT

# 分配设备内存
input_image_gpu = drv.mem_alloc(input_image.nbytes)
output_image_global_gpu = drv.mem_alloc(output_image_global.nbytes)
output_image_texture_gpu = drv.mem_alloc(output_image_texture.nbytes)

# 将输入图像复制到 GPU
drv.memcpy_htod(input_image_gpu, input_image)

# ====== 普通全局内存 Resize ======
block = (16, 16, 1)
grid = ((OUTPUT_WIDTH + block[0] - 1) // block[0], (OUTPUT_HEIGHT + block[1] - 1) // block[1])

start_global = time.time()
resize_global(input_image_gpu, output_image_global_gpu,
              np.int32(INPUT_WIDTH), np.int32(INPUT_HEIGHT),
              np.int32(OUTPUT_WIDTH), np.int32(OUTPUT_HEIGHT),
              block=block, grid=grid)
drv.Context.synchronize()
end_global = time.time()

drv.memcpy_dtoh(output_image_global, output_image_global_gpu)

# ====== 使用纹理内存 Resize ======
# 创建 CUDA 纹理内存
texref = mod_texture.get_texref("tex")
drv.matrix_to_texref(input_image, texref, order="C")
texref.set_address_mode(0, drv.address_mode.CLAMP)
texref.set_address_mode(1, drv.address_mode.CLAMP)
texref.set_filter_mode(drv.filter_mode.LINEAR)

start_texture = time.time()
resize_texture(output_image_texture_gpu, np.int32(OUTPUT_WIDTH), np.int32(OUTPUT_HEIGHT),
               np.float32(scale_x), np.float32(scale_y),
               block=block, grid=grid)
drv.Context.synchronize()
end_texture = time.time()

drv.memcpy_dtoh(output_image_texture, output_image_texture_gpu)

# ====== 打印耗时 ======
print(f"Global Memory Resize Time: {end_global - start_global:.6f} seconds")
print(f"Texture Memory Resize Time: {end_texture - start_texture:.6f} seconds")

# 清理 GPU 内存
drv.Context.synchronize()
input_image_gpu.free()
output_image_global_gpu.free()
output_image_texture_gpu.free()