/*
 * @Author: lexcalibur
 * @Date: 2021-12-07 09:51:50
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-07 13:42:35
 */
#pragma once

// #include "argsParser.h"
// #include "buffers.h"
// #include "parserOnnxConfig.h"
// #include "config.h"

#include "samples/common/argsParser.h"
#include "samples/common/common.h"
#include "samples/common/buffers.h"
#include "samples/common/parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <memory>
#include "common/trt_params.h"
#include "common/logger.h"

class MyNet
{
    template <typename T>
    // using SampleUniquePtr = std::unique_ptr<T, MyNet::InferDeleter>;
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    MyNet(const TRTParams& params)
            : mParams(params)
            , mEngine(nullptr)
    {
    }
    bool build();
    void exec(float* input);

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    TRTParams mParams;
    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;
    int mNumber{0};

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser);
};
