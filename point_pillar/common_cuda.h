#pragma once
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "cuda_runtime_api.h"

#define NUM_THREADS_MACRO 64 // need to be changed when NUM_THREADS is changed

#define DIVUP(m, n) ((m)/(n)+((m)%(n)>0))

#define GPU_CHECK(ans) { GPUAssert((ans), __FILE__, __LINE__); }

inline void GPUAssert(cudaError_t code, const char *file, 
                      int line, bool abort = true)
{
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
    if(abort)
        exit(code);
    }
};