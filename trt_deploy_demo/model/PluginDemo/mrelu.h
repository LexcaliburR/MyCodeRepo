/*
 * @Author: lexcalibur
 * @Date: 2021-12-09 20:00:44
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-09 21:15:24
 */
#pragma once

#include <cuda.h>
#include "NvInferPlugin.h"

#include <cublasLt.h>
#include <string>
#include <vector>



class MReLu : public nvinfer1::IPluginV2DynamicExt
{
public:
    // A.constructor
    // 如果插件有bias或者weight，则需要传入并通过成员变量保留下来
    // 构造函数通常有三个
    // constructor 1, 初始化该插件时的构造函数
    // constructor 2, 用于clone阶段，复制这个构造函数时被调用的函数(理解有问题)
    MReLu(
        const std::string name, const nvinfer1::DataType type, const nvinfer1::Weights& bias);
    // constructor 3, 用于deserialize阶段，用于将序列化好的权重和参数传入该plugin并创建实例
    MReLu(const std::string name, const void* data, size_t length);
    // 无输入的构造函数没用，所以删除
    MReLu() = delete;
    
    // B. Deconstructor
    // 析构函数需要执行terminate，terminate函数释放这个op之前开辟的一些显存空间
    ~MReLu() {  terminate(); };

    // C.copy
    // IPluginV2DynamicExt Methods，这玩意儿干嘛的，顾名思义，就是克隆嘛，将这个plugin对象克隆一份给TensorRT的builder、
    // network或者engine. 调用拷贝构造函数，成员函数主要用于传递不变的权重和参数，将plugin复制n多份，从而可以被不同engine或者builder或者network使用。
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // TensorRT支持Dynamic-shape的时候，batch这个维度必须是explicit的，TensorRT处理的数据形状编委[B,C,W,H]
    // 这个成员函数根据输入维度推理出模型的输出维度，如果插件的输出维度通过世纪运行计算得到，那么这个参数无法得到
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    
    // TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型.
    // 如果插件支持inOut[pos]处的格式/数据类型，则返回true. 如果是否支持取决于其他的输入/输出格式/数据类型，
    // 则插件可以使其结果取决于inOut[0..pos-1]中的格式/数据类型，该格式/数据类型将设置为插件支持的值. 
    // 这个函数不需要检查inOut[pos + 1..nbInputs + nbOutputs-1]，pos的决定必须仅基于inOut[0..pos].
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    // 配置这个插件op，判断输入和输出类型数量是否正确。官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。
    // 但自动调优目前还没有尝试过，我们一般自己写的plugin执行代码都是定死的，所谓的调优步骤可能更多地针对官方的op。
    // 下面的plugin中configurePlugin函数仅仅是简单地确认了下输入和输出以及类型。
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    
    // 这个函数需要返回这个插件op需要中间显存变量的实际数据大小(bytesize)，这个是通过TensorRT的接口去获取，是比较规范的方式。
    // 我们需要在这里确定这个op需要多大的显存空间去运行，在实际运行的时候就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间。
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    
    // 插件运行时调用的执行函数，自己实现的cuda/cpp操作就放在这里，与普通函数一样接受输入，产生输出
    // 注意: 默认写的.cu是fp32的，TensorRT在fp16运行模式下，运行到不支持fp16的插件时，会自动切换到fp32模式，等插件op运行完再切换回来
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods，返回结果的类型，一般来说我们插件op返回结果类型与输入类型一致
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
        noexcept override;

    // IPluginV2 Methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    // 该插件返回多少个tensor,对于的mrelu op只有一个输出，因此实现时返回1
    int getNbOutputs() const noexcept override;

    // 在这个插件准备开始run之前执行，主要时初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行conv,需要提前开辟weight和bias的显存)，
    // 又如该算子运行时需要一些特定的参数，也需要开辟对应的显存空间
    int initialize() noexcept override;

    // 释放这个op之前开辟的一些显存空间
    void terminate() noexcept override;

    // 返回序列化时需要写多少字节到buffer中
    size_t getSerializationSize() const noexcept override;

    // 把需要用的数据按照顺序序列化到buffer里头
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

    // 如函数名，如果不设置默认为""
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // 如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle
    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
    void detachFromContext() noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;

    nvinfer1::DataType mType;
    size_t mNumParams;
    int mNmax;
    int mK;

    bert::WeightsWithOwnership mW;
    bert::cuda_unique_ptr<void> mWdev;
};

class FCPluginDynamicCreator : public nvinfer1::IPluginCreator
{
public:
    // 创建一个空的mPluginAttributes初始化mFC
    FCPluginDynamicCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

    // 这个成员函数作用是通过PluginFieldCollection去创建plugin，将op需要的权重和参数一个一个取出来，然后调用上文提到的第一个构造函数
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    // 这个函数会被onnx-tensorrt的一个叫做TRT_PluginV2的转换op调用，这个op会读取onnx模型的data数据将其反序列化到network中
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    // 这个是成员变量，也会作为getFieldNames成员函数的返回类型。PluginFieldCollection的主要作用是传递这个插件op所需要的权重和参数，
    // 在实际的engine推理过程中并不使用，而在parse中会用到(例如caffe2trt、onnx2trt)。
    // 当使用这些parse去解析这个op的时候，这个op的权重和参数会经历Models --> TensorRT engine --> TensorRT runtime这个过程。
    // 举个例子，在onnx-tensorrt中，我们用过DEFINE_BUILTIN_OP_IMPORTER去注册op，然后通过parse解析onnx模型，
    // 根据注册好的op去一个个解析构建模型
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};


/*
举个例子，在onnx-tensorrt中，我们用过DEFINE_BUILTIN_OP_IMPORTER去注册op，然后通过parse解析onnx模型，
根据注册好的op去一个个解析构建模型，假如我们定义的op为my_custom_op,在DEFINE_BUILTIN_OP_IMPORTER(my_custom_op)会这样实现：

DEFINE_BUILTIN_OP_IMPORTER(mycustom_op)
{
    ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE); 
    ...
    const std::string pluginName = "CUSTOM-OP";
    const std::string pluginVersion = "001";

    // 这个f保存这个op需要的权重和参数，从onnx模型中获取
    std::vector<nvinfer1::PluginField> f;
    // (arg, point, datatype, size)
    f.emplace_back("in_channel", &in_channel, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("weight", kernel_weights.values, nvinfer1::PluginFieldType::kFLOAT32, kernel_weights.count());
    f.emplace_back("bias", bias_weights.values, nvinfer1::PluginFieldType::kFLOAT32, bias_weights.count);

    // 这个从将plugin工厂中获取该插件，并且将权重和参数传递进去
    nvinfer1::IPluginV2* plugin = importPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);

    RETURN_FIRST_OUTPUT(ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin));
}


进入importPluginFromRegistry函数内部，可以发现参数通过fc变量通过createPlugin传递给了plugin：

nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName,
    const std::vector<nvinfer1::PluginField>& pluginFields)
{
    const auto mPluginRegistry = getPluginRegistry();
    const auto pluginCreator
        = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "ONNXTRT_NAMESPACE");

    if (!pluginCreator)
    {
        return nullptr;
    }
    // 接受传进来的权重和参数信息 传递给plugin
    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return pluginCreator->createPlugin(nodeName.c_str(), &fc);
}
上述步骤中，会提供pluginName和pluginVersion初始化MyCustomPluginCreator，其中createPlugin成员函数是我们需要编写的
*/