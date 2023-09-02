/*
 * @Author: lexcalibur
 * @Date: 2023-03-04 20:03:40
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2023-03-04 20:15:11
 */
#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>


// 测试函数，对单例进行添加数据、打印数据的操作，返回执行时间
template <typename Singleton>
std::chrono::milliseconds TestFunction(int thread_num)
{
    std::vector<std::thread> threads;
    std::atomic<int> count{0};
    std::chrono::system_clock::time_point start_time =
        std::chrono::system_clock::now();

    for (int i = 0; i < thread_num; i++) {
        threads.emplace_back([&count]() {
            Singleton::getInstance().AddData(1);
            count++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::chrono::system_clock::time_point end_time =
        std::chrono::system_clock::now();

    if (count != thread_num) {
        std::cout << "Error: count(" << count << ") != thread_num("
                  << thread_num << ")" << std::endl;
    }

    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                 start_time);
}