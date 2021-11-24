cmake_minimum_required(VERSION 3.5)

project(rotation_nms)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(box_iou_rotated)
# add_subdirectory(rotated_nms)
add_subdirectory(test)