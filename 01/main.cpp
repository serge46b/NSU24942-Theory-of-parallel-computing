#include <math.h>

#include <iostream>
#include <vector>

#ifdef USE_DOUBLE
#define ARR_TYPE double
#else
#define ARR_TYPE float
#endif

#define ELEMENT_AMOUNT 10000000

int main() {
  std::cout << "ARR_TYPE: "
            << (std::is_same<ARR_TYPE, double>::value ? "double" : "float")
            << std::endl;
  std::vector<ARR_TYPE> res_arr;
  std::cout << "Calculation started..." << std::endl;
  ARR_TYPE elem_sum = 0;
  for (int i = 0; i < ELEMENT_AMOUNT; i++) {
    ARR_TYPE elem = (ARR_TYPE)sin(i * 2 * 3.14159265359 / ELEMENT_AMOUNT);
    res_arr.push_back(elem);
    elem_sum += elem;
  }
  std::cout << "Sum: " << elem_sum << std::endl;
}