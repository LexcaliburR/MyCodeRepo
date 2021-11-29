/*
 * @Author: lishiqi
 * @LastEditors: lexcaliburr
 */
#pragma once

#include <string>
#include <fstream>
#include "cuda/common_cuda.h"
#include "cuda_runtime_api.h"

template<typename T>
void cudaData2TXT(std::string filename, T* data, int size) {
    T* cpu_data = (T*)malloc(size);
    GPU_CHECK(cudaMemcpy(cpu_data, data, size, cudaMemcpyDeviceToHost));
    std::ofstream out;
    out.open(filename, std::ios::out);
    if(out.is_open()) {
        for(int i = 0; i < size / sizeof(T); i++) {
            out << cpu_data[i] << " ";
        }
    }
    std::cout << "\n";
    out.close();
    return;
}

template<typename T>
void cpuData2TXT(std::string filename, T* data, int size) {
    std::ofstream out;
    out.open(filename, std::ios::out);
    if(out.is_open()) {
        for(int i = 0; i < size / sizeof(T); i++) {
            out << data[i] << " ";
        }
    }
    std::cout << "\n";
    out.close();
    return;
}