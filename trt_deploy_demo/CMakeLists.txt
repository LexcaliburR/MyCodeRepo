cmake_minimum_required(VERSION 3.5)

project(hello_deploy)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "-std=c++14 -g")

set(TRT_PATH /home/library/TensorRT-7.2.3.4)
# set(LOGGER_PATH ${TRT_PATH}/samples/common)

find_package(CUDA REQUIRED)

include_directories(/usr/include
                    # /usr/include/eigen3
                    # ${LOGGER_PATH}
                    ${TRT_PATH}
                    /usr/local/cuda/include
                    ${TRT_PATH}/include
                    ${CUDA_INCLUDE_DIR}
                    /opt/ros/melodic/include
                    ${PROJECT_SOURCE_DIR})

# CUDA_add_executable(${PROJECT_NAME}
#     model/mynet/vergross.cpp
#     model/mynet/vergross.cu
#     main.cpp
#     model/mynet/myNet.cpp
#     common/logger.cpp
# )

CUDA_add_executable(${PROJECT_NAME}
    runPluginDemo.cpp
    common/logger.cpp
    model/PluginDemo/PluginDemo.cpp
)

target_link_libraries(${PROJECT_NAME}
                      ${TRT_PATH}/lib/libnvinfer_plugin.so
                      ${TRT_PATH}/lib/libnvcaffe_parser.so
                      ${TRT_PATH}/lib/libnvonnxparser.so
                      ${TRT_PATH}/lib/libnvparsers.so
                      ${TRT_PATH}/lib/libnvinfer.so
                    #   ${OpenCV_LIBS}
                      ${CUDA_LIBRARIES}
                      ${CUDA_cublas_LIBRARY})