/*
 * @Author: lexcalibur
 * @Date: 2021-11-22 19:38:03
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-11-22 19:57:25
 */
#include <iostream>

class exPerson {
public:
    explicit exPerson(int age)
        :
        age_(age) {};
    void age() { std::cout << "age: " << age_ << std::endl; }
private:
    int age_;
};

class imPerson {
public:
    imPerson(int age)
        :
        age_(age) {};
    imPerson(const imPerson& rhs) { std::cout << "copy assignment func is called" << std::endl; }
    void age() { std::cout << "age: " << age_ << std::endl; }
private:
    int age_;
};

void Print(exPerson p) {
    p.age();
}

void Print2(imPerson p) {
    p.age();
}

int main() {
    // exPerson p1 = 3; // error, 会自动调用构造函数, int(3) -> exPerson(3)
    imPerson p2 = 2; // right, 此时不会调用copy 构造函数
    imPerson p3 = imPerson(3); // right， 不会调用拷贝构造函数
    imPerson px = p2; // right 不会调用拷贝构造函数

    p2.age();
    p3.age();
    px.age();

    exPerson p4 = exPerson(4);
    Print(p4); //right
    // Print(4); // error, 会从 int -> exPerson, 但是exPerson的构造函数为显式的

    Print2(p2); // right
    Print2(2); // right

    return 0;
}