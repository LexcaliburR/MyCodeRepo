/*
 * @Author: lexcaliburr 289427380@gmail.com
 * @Date: 2024-04-19 13:52:05
 * @LastEditors: lexcaliburr 289427380@gmail.com
 * @LastEditTime: 2024-04-19 14:08:18
 * @FilePath: /MyCodeRepo/scheduler/cycler_scheduler.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <atomic>



class CycleScheduler
{
public:
    CycleScheduler() : stop_flag_(false){};
    ~CycleScheduler() { Stop(); };

    CycleScheduler(const CycleScheduler&) = delete;
    CycleScheduler& operator=(const CycleScheduler&) = delete;
    CycleScheduler(CycleScheduler&&) = delete;
    CycleScheduler& operator=(CycleScheduler&&) = delete;

    template <typename F, typename... Args>
    void Start(F&& f, unsigned int interval, Args&&... args)
    {
        Stop();
        stop_flag_ = false;
        cycle_thread_ = std::thread([this, f, interval, args...]() {
            while (!stop_flag_) {
                auto start = std::chrono::high_resolution_clock::now();
                f(args...);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::this_thread::sleep_for(std::chrono::milliseconds(interval) - duration);
            }
        });
    }
    void Stop()
    {
        stop_flag_ = true;
        if (cycle_thread_.joinable()) {
            cycle_thread_.join();
        }
    };

private:
    std::atomic<bool> stop_flag_;
    std::thread cycle_thread_;
};


void PrintFunc() {
    static int i = 0;
    std::cout << "Call " << i++ << " times " << std::endl;
    return;
}

void PrintFunc2(std::string mask) {
    static int i = 0;
    std::cout << "[" << mask << "]" << " Call " << i++ << " times " << std::endl;
    return;
}

class TestClass {
public:
    void PrintFunc() {
        static int i = 0;
        std::cout << "TestClassPrintCall " << i++ << " times " << std::endl;
        return;
    }

    void PrintFunc(std::string mask) {
        static int i = 0;
        std::cout << "[" << mask << "]" << " TestClassPrintCall " << i++ << " times " << std::endl;
        return;
    }
};


int main() {
    CycleScheduler test_scheduler;
    test_scheduler.Start(PrintFunc, 1000);

    CycleScheduler test_scheduler2;
    test_scheduler2.Start(PrintFunc2, 1000, "scheduler2");

    CycleScheduler test_scheduler3;
    TestClass test_class;
    auto boundfunc = std::bind(static_cast<void(TestClass::*)(void)>(&TestClass::PrintFunc), &test_class);
    auto boundfunc2 = std::bind(static_cast<void(TestClass::*)(std::string)>(&TestClass::PrintFunc), &test_class, "boundfunc2");

    test_scheduler3.Start(std::bind(boundfunc, &test_class), 1000);

    CycleScheduler test_scheduler4;
    test_scheduler4.Start(std::bind(boundfunc2, &test_class), 1000);

    
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return 0;   
}