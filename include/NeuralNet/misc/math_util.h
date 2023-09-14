#pragma once

#include <cmath>

namespace NeuralNet::MathUtil {

  template<typename T>
  T Dot(const T *vec1, const T *vec2, size_t size) {
    T dotProduct = T(0);

    for (size_t i = 0; i < size; ++i) {
      dotProduct += vec1[i] * vec2[i];
    }

    return dotProduct;
  }

}