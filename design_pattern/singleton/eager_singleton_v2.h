/*
 * @Author: lexcalibur
 * @Date: 2023-03-04 17:07:49
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2023-03-04 20:17:34
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>

class EagerSingletonV2
{
public:
    // 删除的函数应该是public的，可以通过这样的方式来获得更好的错误信息
    EagerSingletonV2(const EagerSingletonV2 &) = delete;
    EagerSingletonV2 &operator=(const EagerSingletonV2 &) = delete;

public:
    static EagerSingletonV2 &getInstance() { return instance_; }
    void AddData(int num)
    {
        std::lock_guard<std::mutex> guard(m_);
        thread_num_ = num;
        data.push_back(num);
    }
    const int get_thread_num() { return thread_num_; }
    const void PrintData()
    {
        std::lock_guard<std::mutex> guard(m_);
        for (auto &i : data) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }

private:
    int thread_num_ = 0;
    std::vector<int> data;
    EagerSingletonV2() = default;
    std::mutex m_;

    static EagerSingletonV2 instance_;
};

EagerSingletonV2 EagerSingletonV2::instance_;

void SetThreadNum(EagerSingletonV2 &instance, int num)
{
    while (1) {
        instance.AddData(num);
    }
}

void PrintThreadNum(EagerSingletonV2 &instance)
{
    while (1) {
        instance.PrintData();
    }
}

// void MultThreadTestSingletonV2()
// {
//     std::thread t1(SetThreadNum, std::ref(EagerSingletonV2::getInstance()), 1);
//     std::thread t2(SetThreadNum, std::ref(EagerSingletonV2::getInstance()), 3);

//     std::thread t5(PrintThreadNum, std::ref(EagerSingletonV2::getInstance()));
//     std::thread t6(PrintThreadNum, std::ref(EagerSingletonV2::getInstance()));

//     t1.join();
//     t2.join();
//     t5.join();
//     t6.join();
// };

// int main(int argc, char **argv)
// {
//     MultThreadTestSingletonV2();
//     return 0;
// }