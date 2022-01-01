#ifndef SPHERE_H
#define SPHERE_H

#include "sceneobject.h"
#include "vec3.h"

class sphere: public sceneobject {
    public:
        point3 center;
        float radius;

        __device__ sphere() {}
        __device__ sphere(point3 cen, float r): center(cen), radius(r) {};
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;

    // Compute discrim
    float discrim = half_b*half_b - a*c;

    // No solution
    if (discrim < 0) return false;

    float sqrt_disc = sqrt(discrim);

    float root_1 = (-half_b - sqrt_disc) / a;

    if (root_1 < t_max && root_1 > t_min){
        rec.t = root_1;
        rec.p = r.compute_at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
    }

    float root_2 = (-half_b + sqrt_disc) / a;

    if (root_2 < t_max && root_2 > t_min) {
        rec.t = root_2;
        rec.p = r.compute_at(rec.t);
        rec.normal = (rec.p - center) / radius;
        return true;
    }

    return false;

}

#endif