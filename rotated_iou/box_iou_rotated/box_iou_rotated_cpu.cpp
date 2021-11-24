#include <vector>

#include "box_iou_rotated.h"
#include "box_iou_rotated_utils.h"

// template <typename T>
void box_iou_rotated_cpu(
    const std::vector<Box>& boxes1,
    const std::vector<Box>& boxes2,
    std::vector<float>& ious) {
    auto num_boxes1 = boxes1.size();
    auto num_boxes2 = boxes2.size();

    ious.resize(num_boxes1 * num_boxes2);
    for (int i = 0; i < num_boxes1; i++) {
        for (int j = 0; j < num_boxes2; j++) {
            ious[i * num_boxes2 + j] = single_box_iou_rotated<float>(
                boxes1[i], boxes2[j]);
        }
    }
}