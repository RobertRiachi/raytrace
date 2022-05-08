#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "ray.h"
#include "bounding_box.h"
#include "utils.h"

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

class rotate_y : public sceneobject {
    public:
        sceneobject* ptr;
        float sin_theta;
        float cos_theta;
        bool hasbox;
        bounding_box bbox;

        __device__ rotate_y(sceneobject* p, float angle);

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override {
            output_box = bbox;
            return hasbox;
        }
};

__device__ rotate_y::rotate_y(sceneobject* p, float angle) : ptr(p) {
    float radians = degrees_to_rads(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->get_bounding_box(bbox);

    point3 min(p_inf, p_inf, p_inf);
    point3 max(n_inf, n_inf, n_inf);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float x = i*bbox.max().x() + (1-i)*bbox.min().x();
                float y = j*bbox.max().y() + (1-j)*bbox.min().y();
                float z = k*bbox.max().z() + (1-k)*bbox.min().z();

                float new_x = cos_theta*x + sin_theta*z;
                float new_z = -sin_theta*x + cos_theta*z;

                vec3 tester(new_x, y, new_z);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }
    bbox = bounding_box(min, max);
}

__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

    ray rotated_r(origin, direction);

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    auto p = rec.p;
    auto normal = rec.normal;

    p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

    normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;

}

#endif