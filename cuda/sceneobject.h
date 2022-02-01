#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "ray.h"
#include "bounding_box.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    float u;
    float v;
    material *mat_ptr;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class sceneobject {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const = 0;
};

#endif