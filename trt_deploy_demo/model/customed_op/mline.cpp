/*
 * @Author: lexcalibur
 * @Date: 2021-12-14 14:43:04
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-14 16:29:43
 */

#include "torch/script.h"

torch::Tensor mline(torch::Tensor input, torch::Tensor weight) {
    torch::Tensor ret = input.matmul(weight);
    return ret;
}

TORCH_LIBRARY(my_ops, m) {
    m.def("mline", mline);
}