#ifndef MATERIAL_H
#define MATERIAL_H

#include <limits>
#include "ray.h"
#include "sceneobject.h"
#include "vec3.h"
#include "texture.h"

struct hit_record;

class material {
    public:
        __device__ virtual color emitted(float u, float v, const point3& p) const { return color(0,0,0); }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
    public:
        custom_texture* albedo;

        __device__ lambertian(const color& a) : albedo(new rgb_color(a)) {}
        __device__ lambertian(custom_texture *a): albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
                auto scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);
                scattered = ray(rec.p, scatter_direction);
                attenuation = albedo->value(rec.u, rec.v, rec.p);
                return true;
            }
};

class metal : public material {
    public:
        color albedo;
        float fuzz;

        __device__ metal(const color& a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
};

// Schlick approximation for reflectance
__device__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

class dielectric : public material {
    public:
        // Index of refraction
        float ir;

        __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *local_rand_state) const override {
            attenuation = color(1.0, 1.0, 1.0);
            vec3 direction;

            float refraction_ratio = rec.front_face ? (1.0f/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());

            float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

            bool total_internal_reflect = refraction_ratio * sin_theta > 1.0f;

            if (total_internal_reflect || reflectance(cos_theta, refraction_ratio) > curand_uniform(local_rand_state)){
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }

            scattered = ray(rec.p, direction);
            return true;

        }
};

class diffuse_light : public material {
    public:
        custom_texture *emit;

        __device__ diffuse_light(custom_texture *a) : emit(a) {}

        __device__ diffuse_light(color c) : emit(new rgb_color(c)) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray&  scattered, curandState *local_rand_state) const override {
            return false;
        }

        __device__ virtual color emitted(float u, float v, const point3& p) const override {
            return emit->value(u,v,p);
        }

};

#endif