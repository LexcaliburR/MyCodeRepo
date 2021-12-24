/*
 * @Author: lexcalibur
 * @Date: 2021-12-09 17:00:35
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-09 17:22:36
 */


#include <iostream>
#include "model/PluginDemo/PluginDemo.h"
#include "common/trt_params.h"


TRTParams set_params() {
    TRTParams params;
    params.dataDirs.push_back("./");
    params.onnxFileName = "demo.onnx";
    params.inputTensorNames.push_back("input1");
    params.inputTensorNames.push_back("input2");
    params.fp16 = true;
    params.dlaCore = false;
    return params;
}

int main() {
    TRTParams params = set_params();
    PluginDemo model = PluginDemo(params);
    model.build();
    float* input;
    cudaMalloc((void**)(&input), sizeof(float) * 10);
    cudaMemset(input, 9, sizeof(float) * 10);
    model.exec(input);
}