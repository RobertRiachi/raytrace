#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "utils.h"

// Axis Aligned Bounding Box
class bounding_box {
    public:
        point3 minimum;
        point3 maximum;

        __device__ bounding_box() {}
        __device__ bounding_box(const point3& a, const point3& b) { minimum = a; maximum = b;}

        __device__ point3 min() const {return minimum;}
        __device__ point3 max() const {return maximum;}

        __device__ bool hit(const ray& r, float t_min, float t_max) const {
            for (int a = 0; a < 3; a++) {
                auto t0 = fmin((minimum[a] - r.origin()[a]) / r.direction()[a],
                               (maximum[a] - r.origin()[a]) / r.direction()[a]);
                auto t1 = fmax((minimum[a] - r.origin()[a]) / r.direction()[a],
                               (maximum[a] - r.origin()[a]) / r.direction()[a]);
                t_min = fmax(t0, t_min);
                t_max = fmin(t1, t_max);
                if (t_max <= t_min)
                    return false;
            }
            return true;
/*            for(int a = 0; a < 3; a++) {
                float inv_dir = 1.0f / r.direction()[a];
                float t0 = (min()[a] - r.origin()[a]) * inv_dir;
                float t1 = (max()[a] - r.origin()[a]) * inv_dir;

                if (inv_dir < 0.0f) {
                    auto tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }

                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;

                if (t_max <= t_min)
                    return false;
            }
            return true;*/
        }
};

__device__ bounding_box surrounding_box(bounding_box box0, bounding_box box1) {
    point3 small(fmin(box0.min().x(), box1.min().x()),
                 fmin(box0.min().y(), box1.min().y()),
                 fmin(box0.min().z(), box1.min().z()));

    point3 big(fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return bounding_box(small,big);

}

#endif
