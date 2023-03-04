/*
 * @Author: lexcalibur
 * @Date: 2023-03-04 17:07:49
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2023-03-04 20:25:58
 */
#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

class EagerSingletonV1
{
public:
    // 删除的函数应该是public的，可以通过这样的方式来获得更好的错误信息
    EagerSingletonV1(const EagerSingletonV1 &) = delete;
    EagerSingletonV1 &operator=(const EagerSingletonV1 &) = delete;

public:
    static EagerSingletonV1 &getInstance() { return instance_; }
    void AddData(int num)
    {
        thread_num_ = num;
        data.push_back(num);
    }
    const int get_thread_num() { return thread_num_; }
    const void PrintData()
    {
        for (auto &i : data) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

private:
    int thread_num_ = 0;
    std::vector<int> data;
    EagerSingletonV1() = default;

    static EagerSingletonV1 instance_;
};

EagerSingletonV1 EagerSingletonV1::instance_;

void SetThreadNum(EagerSingletonV1 &instance, int num)
{
    while (1) {
        instance.AddData(num);
    }
}

void PrintThreadNum(EagerSingletonV1 &instance)
{
    while (1) {
        instance.PrintData();
    }
}

// void MultThreadTestSingletonV1()
// {
//     std::thread t1(SetThreadNum, std::ref(EagerSingletonV1::getInstance()), 1);
//     std::thread t2(SetThreadNum, std::ref(EagerSingletonV1::getInstance()), 3);

//     std::thread t5(PrintThreadNum, std::ref(EagerSingletonV1::getInstance()));
//     std::thread t6(PrintThreadNum, std::ref(EagerSingletonV1::getInstance()));

//     t1.join();
//     t2.join();
//     t5.join();
//     t6.join();
// };

// int main(int argc, char **argv)
// {
//     MultThreadTestSingletonV1();
//     return 0;
// }