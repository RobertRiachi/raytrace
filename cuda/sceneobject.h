#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "ray.h"

struct hit_record {
    point3 p;
    vec3 normal;
    float t;
};

class sceneobject {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif