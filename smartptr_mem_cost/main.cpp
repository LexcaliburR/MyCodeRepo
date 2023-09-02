/*
 * @Author: lexcalibur
 * @Date: 2022-11-13 11:43:17
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-13 12:05:42
 */
#include <iostream>
#include <memory>

class Doudble8
{
public:
    Doudble8() {}

private:
    double d0;
    double d1;
    double d2;
    double d3;
    double d4;
    double d5;
    double d6;
    double d7;
};

class Double64
{
public:
    Double64() {}

private:
    Doudble8 d0;
    Doudble8 d1;
    Doudble8 d2;
    Doudble8 d3;
    Doudble8 d4;
    Doudble8 d5;
    Doudble8 d6;
    Doudble8 d7;
};

class Double512
{
public:
    Double512() {}

private:
    Double64 d0;
    Double64 d1;
    Double64 d2;
    Double64 d3;
    Double64 d4;
    Double64 d5;
    Double64 d6;
    Double64 d7;
};

class Double1024
{
public:
    Double1024() {}

private:
    Double512 d0;
    Double512 d1;
};

class Double4096
{
public:
    Double4096() {}

private:
    Double1024 d0;
    Double1024 d1;
    Double1024 d3;
    Double1024 d4;
};

class TestClass
{
public:
    TestClass() {}

private:
    Double1024 a0;
    Double1024 a1;
    Double1024 a2;
    Double1024 a3;
    Double1024 a4;
    Double1024 a5;
    Double1024 a6;
    Double1024 a7;
    Double1024 a8;
    Double1024 a9;
    Double1024 a10;
    Double1024 a11;
    Double1024 a12;
    Double1024 a13;
    Double1024 a14;
    Double1024 a15;
    Double1024 a16;
    Double1024 a17;
};

class TestClass2
{
public:
    TestClass2() {}

private:
    TestClass a0;
    TestClass a1;
    TestClass a2;
    TestClass a3;
};

class TestClass3
{
public:
    TestClass3() {}

private:
    TestClass2 a0;
    TestClass2 a1;
    TestClass2 a2;
    TestClass2 a3;
};

class TestClass4
{
public:
    TestClass4() {}

private:
    TestClass3 a0;
    TestClass3 a1;
    TestClass3 a2;
    TestClass2 a4;
    TestClass2 a5;
    Double4096 d0;
    Double4096 d1;
    Double4096 d2;
    Double1024 c0;
    Double1024 c1;
    // Double1024 c1;
};

int main(int argc, char** argv)
{
    std::shared_ptr<int> a = std::make_shared<int>(3);
    std::unique_ptr<int> b = std::make_unique<int>(4);
    std::cout << sizeof(a) << std::endl;
    std::cout << sizeof(b) << std::endl;
    TestClass4 aa;
    std::cout << sizeof(aa) / 8 << std::endl;
    std::cout << (8388608 - sizeof(aa)) / 8 << std::endl;

    while (1) {
    }

    return 0;
}