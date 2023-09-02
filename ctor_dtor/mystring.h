/*
 * @Author: lexcalibur
 * @Date: 2022-11-13 09:18:22
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-13 10:54:21
 */
#ifndef __MYSTRING_H__
#define __MYSTRING_H__

#include <string.h>
#include <iostream>

namespace test_code {

class MyString
{
public:
    MyString()
    {
        data_ = new char[1];
        *data_ = '\0';
        std::cout << "this is ctor with null " << std::endl;
    }
    MyString(const char* data = 0)
    {
        std::cout << "this is ctor " << std::endl;

        if (data) {
            data_ = new char[strlen(data) + 1];
            strcpy(data_, data);
            std::cout << "this is ctor with data " << std::endl;
        } else {
            data_ = new char[1];
            *data_ = '\0';
            std::cout << "this is ctor with null " << std::endl;
        }
    }
    // MyString(const MyString& data) { data_ = strlen(data); }
    // MyString& operator=(const MyString& data);

    ~MyString() { delete[] data_; };
    char* get_c_str() const { return data_; }

private:
    char* data_;
};

inline std::ostream& operator<<(std::ostream& os, const MyString& data) {
  os << data.get_c_str();
  return os;
}

}  // namespace test_code

#endif