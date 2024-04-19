cmake_minimum_required(VERSION 3.5)
project(glog_signalhandler)

find_package(glog)

add_library(pkg1
    ${PROJECT_SOURCE_DIR}/package1.cpp
)

add_library(pkg2
    ${PROJECT_SOURCE_DIR}/package2.cpp
)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    pkg1
    pkg2
    glog::glog
)