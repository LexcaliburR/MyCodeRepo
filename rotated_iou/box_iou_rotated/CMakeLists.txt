cmake_minimum_required(VERSION 3.5)

project(box_iou_rotated)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

add_library(${PROJECT_NAME}
SHARED
    box_iou_rotated_cpu.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${PROJECT_SOURCE_DIR}
)

add_executable(test22
    ./main.cpp
)

target_link_libraries(test22
box_iou_rotated
)