/*
 * @Author: lexcalibur
 * @Date: 2022-11-13 09:18:30
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-13 10:54:55
 */
#include "mystring.h"

using namespace test_code;

int main(int argc, char** argv)
{
    MyString str1();
    MyString str2("this is str2");
    MyString str3("");
    // MyString str3(str2);
    // Mystring str4;
    // str4 = new MyString("hello");

    std::cout << " ---------- ctor 1 -----------\n" << str1 << std::endl;
    std::cout << " ---------- cp ctor 2 -----------\n" << str2 << std::endl;
    std::cout << " ---------- cp ctor 3 -----------\n" << str3 << std::endl;
    // std::cout << " ---------- assign ctor 4 ------------\n"
    //           << str4 << std::endl;
    return 0;
}