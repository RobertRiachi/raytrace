#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "utils.h"

// Axis Aligned Bounding Box
class bounding_box {
    public:
        point3 minimum;
        point3 maximum;

        __device__ bounding_box() {}
        __device__ bounding_box(const point3& a, const point3& b): minimum(a), maximum(b) {}

        __device__ point3 min() const {return minimum;}
        __device__ point3 max() const {return maximum;}

        __device__ __forceinline__ bool hit(const ray& r, float t_min, float t_max) const {
            for(int a = 0; a < 3; a++) {
                float inv_dir = 1.0f / r.direction()[a];
                float t0 = (min()[a] - r.origin()[a]) * inv_dir;
                float t1 = (max()[a] - r.origin()[a]) * inv_dir;

                if (inv_dir < 0.0f) {
                    std::swap(t0,t1);
                }

                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;

                if (t_max <= t_min)
                    return false;
            }
            return true;
        }
};

__device__ bounding_box surrounding_box(bounding_box box0, bounding_box box1) {
    point3 box0_min = box0.min();
    point3 box1_min = box1.min();
    point3 box0_max = box0.max();
    point3 box1_max = box1.max();

    point3 small_box(fmin(box0_min.x(), box1_min.x()),
                     fmin(box0_min.y(), box1_min.y()),
                     fmin(box0_min.z(), box1_min.z()));

    point3 big_box(fmax(box0_max.x(), box1_max.x()),
                   fmax(box0_max.y(), box1_max.y()),
                   fmax(box0_max.z(), box1_max.z()));

    return bounding_box(small_box, big_box);
}

#endif
