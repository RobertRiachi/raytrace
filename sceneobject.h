#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "ray.h"
#include "utils.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    shared_ptr<material> mat_ptr;
    double t;
    bool front_face;

    inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        // If the ray and the normal are in the same direction, inside face
        // if ray and normal are opposite direction, ray is on outside face

        // positive dot product means facing the same direction, negative means different directions
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal: -outward_normal;
    }
};

class sceneobject {
    public:
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& hit_rec) const = 0;
};


#endif