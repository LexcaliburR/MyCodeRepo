#include <vector>
#include <iostream>

#include "box_iou_rotated.h"
#include "box.h"

// void box_iou_rotated_cpu(
//         const std::vector<Box>& boxes1,
//         const std::vector<Box>& boxes2,
//         std::vector<T>& ious);

int main() {
    Box bx1;
    bx1.cx = 0.5;
    bx1.cy = 0.5;
    bx1.w = 1.0;
    bx1.l = 1.0;
    bx1.a = 0.0;

    Box bx2;
    bx2.cx = 1.0;
    bx2.cy = 0.5;
    bx2.w = 1.0;
    bx2.l = 1.0;
    bx2.a = 30.0;

    std::vector<Box> bxes1;
    std::vector<Box> bxes2;

    bxes1.push_back(bx1);
    bxes1.push_back(bx2);
    bxes2.push_back(bx1);
    bxes2.push_back(bx2);


    std::vector<float> res;
    box_iou_rotated_cpu(bxes1, bxes2, res);

    for(int i = 0; i < res.size(); i++) {
        std::cout << res[i] << std::endl;
    }

}