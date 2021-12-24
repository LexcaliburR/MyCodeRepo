/*
 * @Author: lexcalibur
 * @Date: 2021-12-07 09:50:37
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-09 16:51:49
 */

#include <iostream>
#include "model/mynet/myNet.h"
#include "common/trt_params.h"


TRTParams set_params() {
    TRTParams params;
    params.dataDirs.push_back("./");
    params.onnxFileName = "mynet.onnx";
    params.inputTensorNames.push_back("input");
    params.fp16 = true;
    params.dlaCore = false;
    return params;
}

int main() {
    TRTParams params = set_params();
    MyNet model = MyNet(params);
    model.build();
    float* input;
    cudaMalloc((void**)(&input), sizeof(float) * 10);
    cudaMemset(input, 9, sizeof(float) * 10);
    model.exec(input);
}