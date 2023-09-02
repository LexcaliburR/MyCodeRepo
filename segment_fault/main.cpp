/*
 * @Author: lexcalibur
 * @Date: 2022-11-17 14:42:12
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-17 14:42:12
 */
/*
 * @Author: lexcalibur
 * @Date: 2022-11-17 13:22:57
 * @LastEditors: lexcaliburr
 * @LastEditTime: 2022-11-17 14:40:14
 */
#include <iostream>

void segmen_fault1()
{
  float* data  = nullptr;

  data[4] = 4;
  std::cout << data[4] << std::endl;
}

void segmen_fault2()
{
  float* data  = nullptr;

  data[4] = 4;
  std::cout << data[4] << std::endl;
}
void segmen_fault3()
{
  float* data  = nullptr;

  data[4] = 4;
  std::cout << data[4] << std::endl;
}
void segmen_fault4()
{
  float* data  = nullptr;

  data[4] = 4;
  std::cout << data[4] << std::endl;
}
int main(int argc, char** argv)
{

  segmen_fault2();
  return 0;

}