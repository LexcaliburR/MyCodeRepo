/*
 * @Author: lexcalibur
 * @Date: 2021-12-10 14:09:27
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-14 11:05:50
 */
#include "torch/script.h"

torch::Tensor hswish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    torch::Tensor ret = input.clamp_min(-3) * input / 6 + weight + bias;
    return ret;
}

torch::Tensor mrelu(torch::Tensor input, torch::Tensor bias) {
    torch::Tensor ret = relu(input) + bias;
    return ret.clone();
}

TORCH_LIBRARY(my_ops, m) {
    m.def("hswish", hswish);
    m.def("mrelu", mrelu);
}

static auto registry = torch::RegisterOperators("mynamespace::hswish", &hswish);
registry = torch::RegisterOperators("mynamespace::mrelu", &mrelu);

