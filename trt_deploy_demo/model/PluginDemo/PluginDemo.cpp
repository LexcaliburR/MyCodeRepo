/*
 * @Author: lexcalibur
 * @Date: 2021-12-09 16:56:14
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-09 17:22:05
 */

#include "PluginDemo.h"


bool PluginDemo::build()
{
    if (mParams.loadEngine.size() > 0 && !access(mParams.loadEngine.c_str(), 0))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
        {
            infer->setDLACore(mParams.dlaCore);
        }
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        infer->destroy();
        sample::gLogInfo << "TRT Engine loaded from: " << mParams.loadEngine << std::endl;
        if (!mEngine)
        {
            sample::gLogInfo << "TRT Engine loaded failed" << std::endl;
            return false;
        }
        else
        {
            sample::gLogInfo << "TRT Engine loaded successful" << std::endl;
            return true;
        }
    }

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
            = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    
    if (mParams.saveEngine.size() > 0)
    {
        std::ofstream p(mParams.saveEngine, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = mEngine->serialize();
        ASSERT(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.saveEngine << std::endl;
    }
    

    sample::gLogInfo << "getNbInputs: " << network->getNbInputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs: " << network->getNbOutputs() << " \n" << std::endl;
    sample::gLogInfo << "getNbOutputs Name: " << network->getOutput(0)->getName() << " \n" << std::endl;

    mInputDims = network->getInput(0)->getDimensions();

    mOutputDims = network->getOutput(0)->getDimensions();

    return true;
}

void PluginDemo::exec(float* input) {
    samplesCommon::BufferManager buffers(mEngine);
    static auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    float* dinput = static_cast<float*>(buffers.getDeviceBuffer(mParams.inputTensorNames[0]));
    cudaMemcpy(dinput, input, sizeof(float) * 10, cudaMemcpyDeviceToDevice);
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    buffers.copyOutputToHost();
    float* output = static_cast<float*>(buffers.getHostBuffer("output"));
    std::cout << std::endl;
}

bool PluginDemo::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                         SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                         SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(1_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}