/*
 * @Author: lexcalibur
 * @Date: 2021-11-23 19:16:55
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2021-11-23 19:38:13
 */
#include <iostream>
#include <string>

using String = std::string;

class Person {
public:
    explicit Person(String name) // explicit constructor
        : name_(name) 
        {
        }
    Person(const Person& rhs)  // copy constructor
        : name_(rhs.name_)
        {
            std::cout << "copy !" << std::endl;
        }
    Person& operator=(const Person& rhs) // copy assignment
    {
        std::cout << "copy assignment " << std::endl;
        name_ = rhs.name();
        return *this;
    }
    String name() const { return name_; }       // 对函数进行检查,不允许这个函数修改成员变量
    // String const name() { return name_; }    // !未知,需要以后查资料补充
    // const String name() { return name_; }    // 返回的值为const


private:
    String name_;
};

int main() {
    Person p1("person1");
    Person p2 = p1; // 拷贝构造,看左边的对象是否有声明
    Person p3 = Person("2");
    p3 = p2; // 拷贝赋值, 左边对象没有声明
    std::cout << p3.name() << std::endl;
    return 0;
}