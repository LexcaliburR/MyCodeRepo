cmake_minimum_required(VERSION 3.5)

project(rotated_iou_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

include_directories(
    ${PROJECT_SOURCE_DIR}/..
)

add_executable(${PROJECT_NAME}
    rotated_iou_test.cpp
)

target_link_libraries(${PROJECT_NAME}
    box_iou_rotated
)