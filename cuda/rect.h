#ifndef RECT_H
#define RECT_H

#include "sceneobject.h"

__device__ float OFFSET = 0.0001f;

class xy_rect : public sceneobject {
    public:
        material* mat;
        float x0;
        float x1;
        float y0;
        float y1;
        float k;

        __device__ xy_rect() {}

        __device__ xy_rect(float x0, float x1, float y0, float y1, float k, material* mat): x0(x0), x1(x1), y0(y0), y1(y1), k(k), mat(mat) {};

        __device__ virtual bool hit(const ray&r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override {
            output_box  = bounding_box(point3(x0, y0, k-OFFSET), point3(x1, y1, k+OFFSET));
            return true;
        }
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k - r.origin().z()) / r.direction().z();

    if (t < t_min || t > t_max) {
        return false;
    }

    float ray_x_val = r.origin().x() + t*r.direction().x();
    float ray_y_val = r.origin().y() + t*r.direction().y();

    if ( ray_x_val < x0 || ray_x_val > x1 || ray_y_val < y0 || ray_y_val > y1) {
        return false;
    }

    rec.u = (ray_x_val-x0)/(x1-x0);
    rec.v = (ray_y_val-y0)/(y1-y0);
    rec.t = t;

    vec3 outward_norm = vec3(0,0,1);
    rec.set_face_normal(r, outward_norm);
    rec.mat_ptr = mat;
    rec.p = r.compute_at(t);
    return true;
}

class xz_rect : public sceneobject {
    public:
        material* mat;
        float x0;
        float x1;
        float z0;
        float z1;
        float k;


    __device__ xz_rect() {}

        __device__ xz_rect(float x0, float x1, float z0, float z1, float k, material* mat) : x0(x0), x1(x1), z0(z0), z1(z1), k(k), mat(mat) {};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override {
            output_box = bounding_box(point3(x0, k-OFFSET, z0), point3(x1, k+OFFSET, z1));
            return true;
        }
};

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k-r.origin().y()) / r.direction().y();

    if (t < t_min || t > t_max) {
        return false;
    }

    float ray_x_val = r.origin().x() + t*r.direction().x();
    float ray_z_val = r.origin().z() + t*r.direction().z();

    if ( ray_x_val < x0 || ray_x_val > x1 || ray_z_val < z0 || ray_z_val > z1) {
        return false;
    }

    rec.u = (ray_x_val-x0)/(x1-x0);
    rec.v = (ray_z_val-z0)/(z1-z0);
    rec.t = t;

    vec3 outward_norm = vec3(0,1,0);

    rec.set_face_normal(r, outward_norm);
    rec.mat_ptr = mat;
    rec.p = r.compute_at(t);
    return true;
}

class yz_rect : public sceneobject {
    public:
        material* mat;
        float y0;
        float y1;
        float z0;
        float z1;
        float k;

    __device__ yz_rect() {}

        __device__ yz_rect(float y0, float y1, float z0, float z1, float k, material* mat): y0(y0), y1(y1), z0(z0), z1(z1), k(k), mat(mat) {};

        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool get_bounding_box(bounding_box& output_box) const override {
            output_box = bounding_box(point3(k-OFFSET, y0, z0), point3(k+OFFSET, y1, z1));
            return true;
        }
};

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    float t = (k-r.origin().x()) / r.direction().x();

    if (t < t_min || t > t_max) {
        return false;
    }

    float ray_y_val = r.origin().y() + t*r.direction().y();
    float ray_z_val = r.origin().z() + t*r.direction().z();

    if (ray_y_val < y0 || ray_y_val > y1 || ray_z_val < z0 || ray_z_val > z1) {
        return false;
    }

    rec.u = (ray_y_val-y0)/(y1-y0);
    rec.v = (ray_z_val-z0)/(z1-z0);
    rec.t = t;

    vec3 outward_norm = vec3(1,0,0);

    rec.set_face_normal(r, outward_norm);
    rec.mat_ptr = mat;
    rec.p = r.compute_at(t);
    return true;
}
#endif
