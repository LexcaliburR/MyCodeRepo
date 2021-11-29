#pragma once

namespace cuda {

class PreprocessPointCuda {
public:
    PreprocessPointCuda(
            const int max_num_points_in_pillar, 
            const int max_num_pillars, 
            const int num_in_feature,
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
            const float max_z);
    ~PreprocessPointCuda();
    /**
     * @description: 
     * @param[in] points all points in GPU
     * @param[in] num_points the number of input points
     * @param[out] feature feature
     * @param[out] indices indices
     */    
    void Process(
            float* points, 
            int num_points, 
            // float* d_num_points_per_pillar,
            // int* h_pillar_count,
            // float* d_pillar_point_feature, /*final output*/
            // float* d_pillar_coors, /*final output?*/
            int* d_indices, /*final output, input for CenterPoint*/
            float* out_feature /*final output*/
            );

private:
    const int max_num_points_in_pillar_;
    const int max_num_pillars_;
    const int num_point_feature_;
    const int num_out_feature_;

    // size of pillar/voxel 
    const int grid_x_size_;
    const int grid_y_size_;
    const int grid_z_size_;

    // size of each pillar/voxel
    const float pillar_x_length_;
    const float pillar_y_length_;
    const float pillar_z_length_;
    // valid points range
    const float min_x_;
    const float max_x_;
    const float min_y_;
    const float max_y_;
    const float min_z_;
    const float max_z_;

    float* d_points_;
    float* d_pillar_point_feature_in_coors_;
    int* d_pillar_count_hist_;
    int* d_counter_; // the number of points in pillar
    int* d_pillar_count_;
    float* d_points_mean_;
    int* d_indices_;

    float* d_num_points_per_pillar_; // mid
    float* d_pillar_point_feature_; // mid
    float* d_pillar_coors_; // mid

    int* d_x_coors;
    int* d_y_coors;
};

} // namespace cuda
