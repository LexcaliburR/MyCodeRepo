/*
 * @Author: lexcalibur
 * @Date: 2023-03-04 20:04:53
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2023-03-04 20:35:45
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

#include "test_function.h"
#include "eager_singleton_v1.h"
#include "eager_singleton_v2.h"
#include "eager_singleton_v3.h"

// 测试用例数量
// constexpr int kTestCases = 100000;
constexpr int kTestCases = 10000;

int main()
{
    // 测试饿汉式单例的性能
    std::cout << "Testing EagerSingletonV2..." << std::endl;

    for (int i = 1; i <= 16; i++) {
        std::cout << "Thread number: " << i << std::endl;
        std::chrono::milliseconds elapsed_time =
            TestFunction<EagerSingletonV2>(kTestCases / i);
        std::cout << "Elapsed time: " << elapsed_time.count() << "ms"
                  << std::endl;
    }

    std::cout << std::endl;

    // 测试优化后的饿汉式单例的性能
    std::cout << "Testing EagerSingletonV2..." << std::endl;

    for (int i = 1; i <= 16; i++) {
        std::cout << "Thread number: " << i << std::endl;
        std::chrono::milliseconds elapsed_time =
            TestFunction<EagerSingletonV3>(kTestCases / i);
        std::cout << "Elapsed time: " << elapsed_time.count() << "ms"
                  << std::endl;
    }

    return 0;
}