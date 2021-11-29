#include <math.h>

#include "cuda/preprocess_point_cuda.h"
#include "cuda/common_cuda.h"
#include "tic_toc.h"
#include "cuda2txt.h"

namespace cuda {

///////////////////////////////////////
// TODO
// 1. decouple the kernel function with warpped preprocessor
// 2. clean the code
///////////////////////////////////////

/**
 * @description: 
 * @threads: <<<grid_x_size_, grid_y_size_>>>
 * @param[in] points 点云输入
 * @param[in] num_points 点云点的数量
 * @param[in] num_point_feature 输入点云的特征个数(default: 4, xyzi)
 * @param[in] max_num_points_in_pillar 每个pillar的许用点数
 * @param[in] min_x 有效点的最小x坐标值
 * @param[in] max_x 有效点的最大x坐标值
 * @param[in] min_y 有效点的最小y坐标值
 * @param[in] max_y 有效点的最小y坐标值
 * @param[in] min_z 有效点的最小z坐标值
 * @param[in] max_z 有效点的最小z坐标值
 * @param[in] grid_x_size 生成feature map的x轴方向长度(defautl: 468)
 * @param[in] grid_y_size 生成feature map的y轴方向长度(defautl: 468)
 * @param[in] grid_z_size 生成feature map的z轴方向长度(defautl: 1)
 * @param[in] pillar_x_length pillar/voxel的x向长度，单位m(defautl: 0.32)
 * @param[in] pillar_y_length pillar/voxel的y向长度，单位m(defautl: 0.32)
 * @param[in] pillar_z_length depriciated
 * @param[out] d_pillar_count_hist 每个pillar中点数量的统计
 * @param[out] d_pillar_point_feature_in_coors 
 */
__global__ void make_pillar_histo(
        const float* points, 
        const int num_points,
        const int num_point_feature,
        const int max_num_points_in_pillar,
        const float min_x, 
        const float max_x, 
        const float min_y, 
        const float max_y,
        const float min_z, 
        const float max_z,
        const int grid_x_size, 
        const int grid_y_size, 
        const int grid_z_size, 
        const float pillar_x_length,
        const float pillar_y_length,
        const float pillar_z_length,
        int* d_pillar_count_hist,
        float* d_pillar_point_feature_in_coors
        ) {
    int th_i = threadIdx.x + blockIdx.x * blockDim.x;
    bool z_flag = (points[th_i * num_point_feature + 2] <= max_z) && (points[th_i * num_point_feature + 2] >= min_z);
    if(th_i >= num_points 
            || std::isnan(points[th_i * num_point_feature + 0])
            || std::isnan(points[th_i * num_point_feature + 1])
            || std::isnan(points[th_i * num_point_feature + 2])
            || std::isnan(points[th_i * num_point_feature + 3])) {
        return;
    }

    int x_coord = floor((points[th_i * num_point_feature + 0] - min_x) / pillar_x_length);
    int y_coord = floor((points[th_i * num_point_feature + 1] - min_y) / pillar_y_length);
    if( x_coord >= 0 && x_coord < grid_x_size && y_coord >= 0 && y_coord < grid_y_size && z_flag) {
        int count = atomicAdd(&d_pillar_count_hist[x_coord + y_coord * grid_x_size], 1);
        if(count < max_num_points_in_pillar) {
            int idx = y_coord * grid_x_size * max_num_points_in_pillar * num_point_feature 
                    + x_coord * max_num_points_in_pillar * num_point_feature 
                    + count * num_point_feature;
            d_pillar_point_feature_in_coors[idx + 0] = points[th_i * num_point_feature    ];    // x
            d_pillar_point_feature_in_coors[idx + 1] = points[th_i * num_point_feature + 1];    // y
            d_pillar_point_feature_in_coors[idx + 2] = points[th_i * num_point_feature + 2];    // z
            d_pillar_point_feature_in_coors[idx + 3] = points[th_i * num_point_feature + 3];    // i
        }
    }
}

/**
 * @description: 
 * @threads: <<<grid_x_size_, grid_y_size_>>>
 * @param[in] max_num_pillars 最大输出的pillar数，30000
 * @param[in] d_pillar_count_hist grid中，每个pillar的点的数量的统计
 * @param[out] d_x_coors 每个有效pillar的x坐标
 * @param[out] d_y_coors 每个有效pillar的y坐标
 * @param[out] d_counter 所有有效pillar的计数器
 * @param[out] d_pillar_count 保存下来有效pillar的计数器，min(d_counter, max_num_pillars)
 * @param[out] d_num_points_per_pillar 每个pillar的xyz_mean, max_num_pillar x 3
 * @param[out] indices 每个pillar的xy坐标，centernet需要的最终输出结果
 */
__global__ void make_pillar_index(
        const int grid_x_size, 
        const int grid_y_size, 
        const int max_num_points_in_pillar,
        const int max_num_pillars,
        int* d_x_coors,  /*unknown output*/
        int* d_y_coors,  /*unknown output*/
        int* d_counter, /*所有pillar中有效的pillar数量 output*/
        int* d_pillar_count, /*保存下来的pillar数量 output*/
        float* d_pillar_point_feature_in_coors,
        const int* d_pillar_count_hist,
        float* d_num_points_per_pillar, /*output 对应pillar的point数，用于计算最终输出结果*/
        int* indices) {
    int x = blockIdx.x;
    int y = threadIdx.x;

    int num_points_at_this_pillar = d_pillar_count_hist[y * grid_x_size + x];

    if(num_points_at_this_pillar == 0) {
        return;
    }

    int count = atomicAdd(d_counter, 1);
    if(count < max_num_pillars) {
        atomicAdd(d_pillar_count, 1);
        if(num_points_at_this_pillar >= max_num_points_in_pillar) {
            d_num_points_per_pillar[count] = max_num_points_in_pillar;
        } else {
            d_num_points_per_pillar[count] = num_points_at_this_pillar;
        }
        // 保存下来用来生成输出的每个pillar的下标
        d_x_coors[count] = x;
        d_y_coors[count] = y;
        indices[count * 2 + 0] = -1;
        indices[count * 2 + 1] = y * grid_x_size + x;
    }
}

// h_pillar_count[0], max_num_points_in_pillar_
__global__ void make_pillar_feature(
        const float* d_pillar_point_feature_in_coors,
        const int* d_x_coors,
        const int* d_y_coors,
        const float* d_num_points_per_pillar,
        const int max_num_points_in_pillar,
        const int num_point_feature,
        const int grid_x_size,
        float* d_pillar_point_feature, /*output*/
        float* d_pillar_coors /*output*/) {
    int ith_pillar = blockIdx.x;
    int ith_point = threadIdx.x;
    int num_points_at_this_pillar = d_num_points_per_pillar[ith_pillar];
    if(ith_point >= num_points_at_this_pillar) {
        return;
    }

    int x_idx = d_x_coors[ith_pillar];
    int y_idx = d_y_coors[ith_pillar];

    int pillar_idx = ith_pillar * max_num_points_in_pillar * num_point_feature
                   + ith_point * num_point_feature;
    int coors_idx = y_idx * grid_x_size * max_num_points_in_pillar * num_point_feature
                  + x_idx * max_num_points_in_pillar * num_point_feature
                  + ith_point * num_point_feature;

    d_pillar_point_feature[pillar_idx + 0] = d_pillar_point_feature_in_coors[coors_idx + 0];
    d_pillar_point_feature[pillar_idx + 1] = d_pillar_point_feature_in_coors[coors_idx + 1];
    d_pillar_point_feature[pillar_idx + 2] = d_pillar_point_feature_in_coors[coors_idx + 2];
    d_pillar_point_feature[pillar_idx + 3] = d_pillar_point_feature_in_coors[coors_idx + 3];
    // printf("idx: %d, x: %f\n", pillar_idx, d_pillar_point_feature[pillar_idx + 0]);
}

/**
 * @description: 
 * @threads: <<<h_pillar_num, mean_block(20,3)>>>
 * @param[in] num_point_feature 4:x, y, z, i
 * @param[in] d_pillar_point_feature pillar里20个点每个点的feature
 * @param[in] d_num_points_per_pillar 取出来的所有点的point数，下标为count
 * @param[in] max_num_pillars 最大的pillar数(default 30000)
 * @param[in] max_num_point_per_pillar 每个pillar最大允许点数(default 20)
 * @param[out] d_points_mean 每个pillar的xyz_mean, max_num_pillar x 3
 */
__global__ void pillar_mean_kernel(
        const int num_point_feature,
        const float* d_pillar_point_feature,
        const float* d_num_points_per_pillar,
        const int max_num_pillars,
        const int max_num_point_per_pillar,
        float* d_points_mean) {
    extern __shared__ float temp[];
    int ith_pillar = blockIdx.x; // [0, 1, ... , 30000]
    int ith_point = threadIdx.x; // [0, 1, ... , 20]
    int axis = threadIdx.y; // [0, 1, 2]
    // 对齐
    int reduce_size = max_num_point_per_pillar > 32 ? 64 : 32;
    temp[ith_point * 3 + axis] = d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + axis];

    if(ith_point < reduce_size - max_num_point_per_pillar) {
        temp[(threadIdx.x + max_num_point_per_pillar) * 3 + axis] = 0.0f; 
    }
    __syncthreads();
    int num_points_at_this_pillar = d_num_points_per_pillar[ith_pillar];

    if(ith_point >= num_points_at_this_pillar) {
        return;
    }

    for(unsigned int d = reduce_size >> 1; d > 0.6; d >>= 1) {
        if(ith_point < d) {
            temp[ith_point * 3 + axis] += temp[(ith_point + d) * 3 + axis];
        }
        __syncthreads();
    }

    if(ith_point == 0) {
        d_points_mean[ith_pillar * 3 + axis] = temp[ith_point + axis] / num_points_at_this_pillar;
    }
}

/**
 * @description: depreciated
 * @threads: <<<h_pillar_num, mean_block>>>
 * @param[in] num_point_feature 4:x, y, z, i
 * @param[in] d_pillar_point_feature pillar里20个点每个点的feature
 * @param[in] d_num_points_per_pillar 取出来的所有点的point数，下标为count
 * @param[in] max_num_pillars 最大的pillar数，30000
 * @param[in] max_num_point_per_pillar 每个pillar最大允许点数,20
 * @param[out] d_points_mean 每个pillar的xyz_mean, max_num_pillar x 3
 */
/* depreciated
__global__ void pillar_mean_kernel2(
        const int num_point_feature,
        const float* d_pillar_point_feature,
        const float* d_num_points_per_pillar,
        const int max_num_pillars,
        const int max_num_point_per_pillar,
        float* d_points_mean) {
    
    extern __shared__ float temp[];
    int ith_pillar = blockIdx.x;
    int ith_point = threadIdx.x;
    int num_points_at_this_pillar = d_num_points_per_pillar[ith_pillar];

    for(int i = 0; i < num_points_at_this_pillar; i++) {
        atomicAdd(&temp[i * 3 + 0], d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 0]);
        atomicAdd(&temp[i * 3 + 1], d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 1]);
        atomicAdd(&temp[i * 3 + 2], d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 2]);
        // temp[i * 3 + 0] += d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 0];
        // temp[i * 3 + 1] += d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 1];
        // temp[i * 3 + 2] += d_pillar_point_feature[ith_pillar * max_num_point_per_pillar * num_point_feature + ith_point * num_point_feature + 2];
    }
    __syncthreads();
    d_points_mean[ith_pillar * 3 + 0] = temp[ith_point + 0] / num_points_at_this_pillar;
    d_points_mean[ith_pillar * 3 + 1] = temp[ith_point + 1] / num_points_at_this_pillar;
    d_points_mean[ith_pillar * 3 + 2] = temp[ith_point + 2] / num_points_at_this_pillar;
    __syncthreads();
}
*/

/**
 * @description: <<<max_num_pillars_, max_num_points_per_pillar_>>>
 * @param[in] d_pillar_point_feature
 * @param[in] d_num_points_per_pillar
 * @param[in] d_points_mean
 * @param[in] max_num_pillars
 * @param[in] max_num_points_in_pillars
 * @param[in] num_point_feature 
 * @param[out] out_feature
 */
__global__ void gather_point_feature_kernel(
        const float* d_pillar_point_feature,
        const float* d_num_points_per_pillar,
        const float* d_points_mean,
        const int* d_x_coors,
        const int* d_y_coors,
        const int max_num_pillars,
        const int max_num_points_in_pillars,
        const int num_point_feature,
        const float min_x, 
        const float max_x, 
        const float min_y, 
        const float max_y,
        const float pillar_x_length,
        const float pillar_y_length,
        float* out_feature) {
    int ith_pillar = blockIdx.x;
    int ith_point = threadIdx.x;
    int x_idx = d_x_coors[ith_pillar];
    int y_idx = d_y_coors[ith_pillar];

    int num_points_in_this_pillar = d_num_points_per_pillar[ith_pillar];
    if(num_points_in_this_pillar <= 0 || ith_point >= num_points_in_this_pillar) {
        return;
    }

    out_feature[0*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = d_pillar_point_feature[ith_pillar * max_num_points_in_pillars * num_point_feature + ith_point * num_point_feature + 0]; // x
    out_feature[1*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = d_pillar_point_feature[ith_pillar * max_num_points_in_pillars * num_point_feature + ith_point * num_point_feature + 1]; // y
    out_feature[2*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = d_pillar_point_feature[ith_pillar * max_num_points_in_pillars * num_point_feature + ith_point * num_point_feature + 2]; // z
    out_feature[3*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = d_pillar_point_feature[ith_pillar * max_num_points_in_pillars * num_point_feature + ith_point * num_point_feature + 3]; // i
    out_feature[4*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = 0.1;
    out_feature[5*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = out_feature[0*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point] - d_points_mean[ith_pillar * 3 + 0]; // x - cx
    out_feature[6*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = out_feature[1*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point] - d_points_mean[ith_pillar * 3 + 1]; // y - cy
    out_feature[7*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = out_feature[2*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point] - d_points_mean[ith_pillar * 3 + 2];  // z - cz
    out_feature[8*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = out_feature[0*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        - (x_idx * pillar_x_length + min_x + pillar_x_length / 2.0f); // x_idx * X_STEP + X_MIN + X_STEP / 2
    out_feature[9*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
        = out_feature[1*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point] 
        - (y_idx * pillar_y_length + min_y + pillar_y_length / 2.0f);  // // y_idx * Y_STEP + Y_MIN + Y_STEP / 2
    
    // printf("x: %f, y: %f, z: %f, i: %f, cx: %f, cy: %f, cz: %f, bx: %f, by: %f, bz: %f\n", 
    //     out_feature[0*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[1*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[2*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[3*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[4*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[5*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[6*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[7*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[8*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point],
    //     out_feature[9*max_num_points_in_pillars*max_num_pillars + ith_pillar*max_num_points_in_pillars + ith_point]
    //     );
}

/**
* @brief Constructor
* @param[in] num_threads Number of threads when launching cuda kernel
* @param[in] num_point_feature Number of features in a point
* @param[in] max_num_pillars Maximum number of pillars
* @param[in] max_points_per_pillar Maximum number of points per pillar
*/
PreprocessPointCuda::PreprocessPointCuda(
        const int max_num_points_in_pillar, 
        const int max_num_pillars, 
        const int num_point_feature,
        const int num_out_feature,
        const int grid_x_size, 
        const int grid_y_size, 
        const int grid_z_size,
        const float pillar_x_length,
        const float pillar_y_length,
        const float pillar_z_length,
        const float min_x,
        const float max_x,
        const float min_y,
        const float max_y,
        const float min_z,
        const float max_z): 
        max_num_points_in_pillar_(max_num_points_in_pillar),
        max_num_pillars_(max_num_pillars),
        num_point_feature_(num_point_feature),
        num_out_feature_(num_out_feature),
        grid_x_size_(grid_x_size),
        grid_y_size_(grid_y_size),
        grid_z_size_(grid_z_size),
        pillar_x_length_(pillar_x_length),
        pillar_y_length_(pillar_y_length),
        pillar_z_length_(pillar_z_length),
        min_x_(min_x),
        max_x_(max_x),
        min_y_(min_y),
        max_y_(max_y),
        min_z_(min_z),
        max_z_(max_z) {

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pillar_count_hist_), grid_x_size_ * grid_y_size_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pillar_point_feature_in_coors_), grid_x_size_ * grid_y_size_ * max_num_points_in_pillar_ * num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counter_), sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pillar_count_), sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points_mean_), max_num_pillars_ * 3 * sizeof(float)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x_coors), max_num_pillars_ * sizeof(int)));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y_coors), max_num_pillars_ * sizeof(int)));

    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_num_points_per_pillar_), sizeof(float) * grid_x_size_ * grid_y_size_));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pillar_point_feature_), sizeof(float) * max_num_pillars_ * 4 * max_num_points_in_pillar_));
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pillar_coors_), sizeof(float) * grid_x_size_ * grid_y_size_ * 2));
}

PreprocessPointCuda::~PreprocessPointCuda() {
    GPU_CHECK(cudaFree(d_pillar_point_feature_in_coors_));
    GPU_CHECK(cudaFree(d_pillar_count_hist_));
    GPU_CHECK(cudaFree(d_counter_));
    GPU_CHECK(cudaFree(d_pillar_count_));
    GPU_CHECK(cudaFree(d_points_mean_));
    GPU_CHECK(cudaFree(d_num_points_per_pillar_));
    GPU_CHECK(cudaFree(d_pillar_point_feature_));
    GPU_CHECK(cudaFree(d_pillar_coors_));
    GPU_CHECK(cudaFree(d_x_coors));
    GPU_CHECK(cudaFree(d_y_coors));
}

/**
* @description: 
* @param[mid] d_x_coors X-coordinate indexes for corresponding pillars
* @param[mid] d_y_coors Y-coordinate indexes for corresponding pillars
* @param[mid] dev_num_points_per_pillar, Number of points in corresponding pillars
* @param[out] dev_sparse_pillar_map Grid map representation for pillar-occupancy
*/
void PreprocessPointCuda::Process(
        float* points, 
        int num_points, 
        int* d_indices, /*final output, input for CenterPoint*/
        float* out_feature /*final output*/) {
#ifdef USE_CUDA
PERF_BLOCK_START(true);
#endif
    GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points_), num_points * sizeof(float) * 4));
    GPU_CHECK(cudaMemset(d_points_, 0, num_points * sizeof(float) * 4));
    GPU_CHECK(cudaMemcpy(d_points_, points, num_points * sizeof(float) * 4, cudaMemcpyHostToDevice));
#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] copy point from cpu to gpu");
#endif
    GPU_CHECK(cudaMemset(d_pillar_count_hist_, 0, grid_x_size_ * grid_y_size_ * sizeof(int)));
    GPU_CHECK(cudaMemset(d_pillar_point_feature_in_coors_, 0, grid_x_size_ * grid_y_size_ * max_num_points_in_pillar_ * num_point_feature_ * sizeof(float)));
    GPU_CHECK(cudaMemset(d_counter_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(d_pillar_count_, 0, sizeof(int)));
    GPU_CHECK(cudaMemset(d_points_mean_, 0,  max_num_pillars_ * 3 * sizeof(float)));
    GPU_CHECK(cudaMemset(d_x_coors, 0, max_num_pillars_ * sizeof(int)));
    GPU_CHECK(cudaMemset(d_y_coors, 0, max_num_pillars_ * sizeof(int)));
    GPU_CHECK(cudaMemset(d_num_points_per_pillar_, 0, sizeof(float) * grid_x_size_ * grid_y_size_));
    GPU_CHECK(cudaMemset(d_pillar_point_feature_, 0, sizeof(float) * max_num_pillars_ * 4 * max_num_points_in_pillar_));
    GPU_CHECK(cudaMemset(d_pillar_coors_, -1, sizeof(float) * grid_x_size_ * grid_y_size_ * 2));

#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] variable cudaMemset");
#endif
    int num_blocks = DIVUP(num_points, NUM_THREADS_MACRO);
    // A.make pillar histogram / 统计每个pillar点的数量，将pillar里的点保存下来
    make_pillar_histo<<<num_blocks, NUM_THREADS_MACRO>>>(
        d_points_, 
        num_points,
        num_point_feature_,
        max_num_points_in_pillar_,
        min_x_, 
        max_x_, 
        min_y_, 
        max_y_,
        min_z_, 
        max_z_,
        grid_x_size_,
        grid_y_size_,
        grid_z_size_,
        pillar_x_length_,
        pillar_y_length_,
        pillar_z_length_,
        d_pillar_count_hist_,  /*output*/
        d_pillar_point_feature_in_coors_ /*output*/
    );
    // cudaData2TXT("/workspace/robotaxi/apal-robot-percp/Test/testLidarPipeline/d_pillar_count_hist_.txt", d_pillar_count_hist_, grid_x_size_*grid_y_size_*sizeof(int));
#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] make_pillar_histo");
#endif

    // B.make pillar index / 统计需要保留的pillar的点的数量，以grip map的形式保存有效的pillar的下标
    // warning: make sure grid_y_size smaller than maximal threads in one SM(according to GPU)
    make_pillar_index<<<grid_x_size_, grid_y_size_>>>(
        grid_x_size_,
        grid_y_size_,
        max_num_points_in_pillar_,
        max_num_pillars_,
        d_x_coors,
        d_y_coors,
        d_counter_,
        d_pillar_count_,
        d_pillar_point_feature_in_coors_,
        d_pillar_count_hist_,
        d_num_points_per_pillar_, /*output 对应pillar的point数，用于计算最终输出结果*/
        d_indices
    );

    // cudaData2TXT("/workspace/robotaxi/apal-robot-percp/Test/testLidarPipeline/d_indices.txt", d_indices, max_num_pillars_ * 2 * sizeof(int));

#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] make_pillar_index ");
#endif
    // // C.make pillar feature / 取出每个pillar中所有点的值
    int h_pillar_num = 0;
    GPU_CHECK(cudaMemcpy(&h_pillar_num, d_pillar_count_, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    make_pillar_feature<<<h_pillar_num, max_num_points_in_pillar_>>>(
        d_pillar_point_feature_in_coors_,
        d_x_coors,
        d_y_coors,
        d_num_points_per_pillar_,
        max_num_points_in_pillar_,
        num_point_feature_,
        grid_x_size_,
        d_pillar_point_feature_, /*output*/
        d_pillar_coors_ /*output*/
    );
    // cudaData2TXT("/workspace/robotaxi/apal-robot-percp/Test/testLidarPipeline/d_pillar_feature.txt", d_pillar_point_feature_, 468*468*4*20*sizeof(float));

#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] make_pillar_feature")
#endif
    // // D.compute mean val in outputed pillar
    dim3 mean_block(max_num_points_in_pillar_,3); // (20,3)
    pillar_mean_kernel<<<h_pillar_num, mean_block, 64 * 3 *sizeof(float)>>>(
        num_point_feature_,
        d_pillar_point_feature_,
        d_num_points_per_pillar_,
        max_num_pillars_,
        max_num_points_in_pillar_,
        d_points_mean_
    );

#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] pillar_mean_kernel")
#endif

    gather_point_feature_kernel<<<max_num_pillars_, max_num_points_in_pillar_>>>(
        d_pillar_point_feature_,
        d_num_points_per_pillar_,
        d_points_mean_,
        d_x_coors,
        d_y_coors,
        max_num_pillars_,
        max_num_points_in_pillar_,
        num_point_feature_,
        min_x_, 
        max_x_, 
        min_y_, 
        max_y_,
        pillar_x_length_,
        pillar_y_length_,
        out_feature
    );
    // cudaData2TXT("/workspace/robotaxi/apal-robot-percp/Test/testLidarPipeline/feature.txt", out_feature, 30000*20*10*sizeof(float));

#ifdef USE_CUDA
PERF_BLOCK_END("[Preprocess] gather_point_feature_kernel");
#endif
}

} // namespace cuda