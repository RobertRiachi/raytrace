#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include "vec3.h"

const float pi = 3.1415926535897932385;

inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#endif
