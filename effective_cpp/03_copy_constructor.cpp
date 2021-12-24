/*
 * @Author: lexcalibur
 * @Date: 2021-12-24 17:35:07
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-12-24 17:55:45
 */
#include <iostream>

class A {
public:
    A() {};
    A(A &rhs) { std::cout << "call copy construct func!" << std::endl; };
    ~A() {};
    // A& operator=(A &rhs) { std::cout << "call copy assignment func!" << std::endl; };
    int id;
};

void func1(A a) {}

void func2(A& a) {}

A func3() {  
    A a;
    a.id = 3;
    return a; 
}

A func4(A a) { return a; }

int main() {
    A a1;

    std::cout << "============== case 1 ==================" << std::endl;
    A a2(a1); // 1 time
    std::cout << "============== case 2 ==================" << std::endl;
    func1(a1); // 1 time
    std::cout << "============== case 3 ==================" << std::endl;
    func2(a1); // 0 time
    std::cout << "============== case 4 ==================" << std::endl;
    func3(); // 0 time
    std::cout << "============== case 4 ==================" << std::endl;
    a2 = func3(); // 0 time
    std::cout << "a2 id: " << a2.id << std::endl;
    std::cout << "============== case 5 ==================" << std::endl;
    func4(a1); // 2 times
    std::cout << "============== case 6 ==================" << std::endl;
    a2 = func4(a1); // 2 + 1
}
