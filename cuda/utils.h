#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include "vec3.h"

__device__ const float pi = 3.1415926535897932385;
__device__ const float p_inf = INFINITY;
__device__ const float n_inf = -INFINITY;

float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ __forceinline__ float degrees_to_rads(float degree) { return degree * pi / 180.0f; }

#endif
