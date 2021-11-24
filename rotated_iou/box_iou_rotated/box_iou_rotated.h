#pragma once
#include "box.h"

// template <typename T>
void box_iou_rotated_cpu(
        const std::vector<Box>& boxes1,
        const std::vector<Box>& boxes2,
        std::vector<float>& ious);