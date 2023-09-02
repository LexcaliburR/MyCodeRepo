/*
 * @Author: lexcalibur
 * @Date: 2023-03-04 20:22:32
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2023-03-04 20:34:50
 */
#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

class EagerSingletonV3
{
public:
    EagerSingletonV3(const EagerSingletonV3 &) = delete;
    EagerSingletonV3 &operator=(const EagerSingletonV3 &) = delete;

public:
    static EagerSingletonV3 &getInstance()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (instance_ == nullptr) {
            instance_ = new EagerSingletonV3();
        }
        return *instance_;
    }
    void AddData(int num)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        thread_num_ = num;
        data.push_back(num);
        lock.unlock();
        cv_.notify_all();
    }
    const int get_thread_num() { return thread_num_; }
    const void PrintData()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !data.empty(); });
        for (auto &i : data) {
            std::cout << i << " ";
        }
        std::cout << std::endl;
        data.clear();
    }

private:
    int thread_num_ = 0;
    std::vector<int> data;
    static EagerSingletonV3 *instance_;
    static std::mutex mutex_;
    std::condition_variable cv_;

    EagerSingletonV3() = default;
};

EagerSingletonV3 *EagerSingletonV3::instance_ = nullptr;
std::mutex EagerSingletonV3::mutex_;