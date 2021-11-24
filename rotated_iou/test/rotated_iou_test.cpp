#include <vector>
#include <iostream>
#include <chrono>

#include "box_iou_rotated/box_iou_rotated.h"
#include "box_iou_rotated/box.h"

// void box_iou_rotated_cpu(
//         const std::vector<Box>& boxes1,
//         const std::vector<Box>& boxes2,
//         std::vector<T>& ious);

void test_iou_05_overlap() {
    std::cout << "--------- test iou 05 overlap ----------" << std::endl;
    Box bx1;
    bx1.cx = 0.5;
    bx1.cy = 0.5;
    bx1.w = 1.0;
    bx1.l = 1.0;
    bx1.a = 0.0;

    Box bx2;
    bx2.cx = 0.25;
    bx2.cy = 0.5;
    bx2.w = 0.5;
    bx2.l = 1.0;
    bx2.a = 0.0;

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

void test_iou_precision() {
    std::cout << "--------- test_iou_precision ----------" << std::endl;
    Box bx1;
    bx1.cx = 565;
    bx1.cy = 565;
    bx1.w = 10;
    bx1.l = 10;
    bx1.a = 0.0;

    Box bx2;
    bx2.cx = 565;
    bx2.cy = 565;
    bx2.w = 10;
    bx2.l = 8.3;
    bx2.a = 0.0;

    std::vector<Box> bxes1;
    std::vector<Box> bxes2;

    bxes1.push_back(bx1);
    bxes2.push_back(bx2);


    std::vector<float> res;
    box_iou_rotated_cpu(bxes1, bxes2, res);

    for(int i = 0; i < res.size(); i++) {
        std::cout << res[i] << std::endl;
    }
    std::cout << "right iou " << 8.3 / 10 << std::endl;
}

void test_iou_too_many_boxes() {
    std::cout << "--------- test_iou_too_many_boxes ----------" << std::endl;
    std::vector<Box> bxes1;
    std::vector<Box> bxes2;

    for(int i = 0; i < 1000; i++) {
        Box tmp;
        bxes1.push_back(tmp);
        bxes2.push_back(tmp);
    }

    std::vector<float> res;
    auto start = std::chrono::system_clock::now();
    box_iou_rotated_cpu(bxes1, bxes2, res);
    auto end = std::chrono::system_clock::now();
    // auto duration = static_cast<std::chrono::>

}
    // def test_iou_precision(self):
    //     for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
    //         boxes1 = torch.tensor([[565, 565, 10, 10.0, 0]], dtype=torch.float32, device=device)
    //         boxes2 = torch.tensor([[565, 565, 10, 8.3, 0]], dtype=torch.float32, device=device)
    //         iou = 8.3 / 10.0
    //         expected_ious = torch.tensor([[iou]], dtype=torch.float32)
    //         ious = pairwise_iou_rotated(boxes1, boxes2)
    //         self.assertTrue(torch.allclose(ious.cpu(), expected_ious))

int main() {
    test_iou_05_overlap();
    test_iou_precision();
    test_iou_too_many_boxes();
}