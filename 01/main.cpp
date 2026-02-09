#include <math.h>

#include <iostream>

#ifdef USE_DOUBLE
#define ARR_TYPE double
#else
#define ARR_TYPE float
#endif

#define ELEMENT_AMOUNT 10000000
ARR_TYPE res_arr[ELEMENT_AMOUNT];

int main() {
  std::cout << "ARR_TYPE: "
            << (std::is_same<ARR_TYPE, double>::value ? "double" : "float")
            << std::endl;
  std::cout << "Calculation started..." << std::endl;
  ARR_TYPE elem_sum = 0;
  for (int i = 0; i < ELEMENT_AMOUNT; i++) {
    res_arr[i] = sin(i * 2 * 3.14 / ELEMENT_AMOUNT);
    elem_sum += res_arr[i];
  }
  std::cout << "Sum: " << elem_sum << std::endl;
}