cmake_minimum_required(VERSION 3.5)

project(customed_ops)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "-std=c++14 -g")

set(TRT_PATH /home/library/TensorRT-7.2.3.4)
# set(LOGGER_PATH ${TRT_PATH}/samples/common)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# export Torch_DIR=/home/lishiqi/anaconda3/lib/python3.8/site-packages/torch/share/cmake/Torch

include_directories(/usr/include
                    # /usr/include/eigen3
                    # ${LOGGER_PATH}
                    ${TRT_PATH}
                    /usr/local/cuda/include
                    ${TRT_PATH}/include
                    ${CUDA_INCLUDE_DIR}
                    /opt/ros/melodic/include
                    ${TORCH_INCLUDE_DIRS})

add_library(mrelu
  SHARED
  mrelu_op.cpp
)

target_link_libraries(mrelu
                      ${TORCH_LIBRARIES}
                      # ${TRT_PATH}/lib/libnvinfer_plugin.so
                      # ${TRT_PATH}/lib/libnvcaffe_parser.so
                      # ${TRT_PATH}/lib/libnvonnxparser.so
                      # ${TRT_PATH}/lib/libnvparsers.so
                      # ${TRT_PATH}/lib/libnvinfer.so
                    #   ${OpenCV_LIBS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_cublas_LIBRARY}
                      ${TORCH_LIBRARIES})

add_library(hswish
  SHARED
  hswish_op.cpp
)

target_link_libraries(hswish
                      ${TORCH_LIBRARIES}
                      # ${TRT_PATH}/lib/libnvinfer_plugin.so
                      # ${TRT_PATH}/lib/libnvcaffe_parser.so
                      # ${TRT_PATH}/lib/libnvonnxparser.so
                      # ${TRT_PATH}/lib/libnvparsers.so
                      # ${TRT_PATH}/lib/libnvinfer.so
                    #   ${OpenCV_LIBS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_cublas_LIBRARY}
                      ${TORCH_LIBRARIES})