/*
 * @Author: lexcalibur
 * @Date: 2021-12-07 10:04:12
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-07 10:46:42
 */
#pragma once
#include <vector>
#include <string>

struct TRTParams
{
    int32_t batchSize{1};              //< Number of inputs in a batch
    int32_t dlaCore{-1};               //< Specify the DLA core to run network on.
    bool int8{false};                  //< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFileName;
    std::string loadEngine;
    std::string saveEngine;
    bool gpu_preprocess{true};
};
