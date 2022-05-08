#ifndef SPHERE_H
#define SPHERE_H

#include "sceneobject.h"
#include "vec3.h"
#include "utils.h"


class sphere: public sceneobject {
    public:
        point3 center;
        float radius;
        material *mat_ptr;

    __device__ sphere() {}
        __device__ sphere(point3 cen, float r, material *m): center(cen), radius(r), mat_ptr(m) {};
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override;

        __device__ static void get_sphere_uv(const point3 &p, float &u, float &v) {
            float theta = acos(-p.y());
            float phi = atan2(-p.z(), p.x()) + pi;

            u = phi / (2*pi);
            v = theta / pi;
        }

};

__device__ bool sphere::get_bounding_box(bounding_box &output_box) const {
    output_box = bounding_box(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}

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
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    float root_2 = (-half_b + sqrt_disc) / a;

    if (root_2 < t_max && root_2 > t_min) {
        rec.t = root_2;
        rec.p = r.compute_at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    return false;

}

#endif