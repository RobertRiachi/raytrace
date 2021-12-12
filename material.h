#ifndef MATERIAL_H
#define MATERIAL_H

#include "utils.h"

struct hit_record;

class material {
    public:
        virtual bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered
        ) const = 0;
};

class lambertian : public material {
    public:
        color albedo;

        lambertian(const color& a) : albedo(a) {}

        virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
            vec3 scattered_dir = rec.normal + random_unit_vector();

            // To avoid divide by 0 errors if result is near zero replace with normal vec for computational stability
            if (scattered_dir.near_zero()){
                scattered_dir = rec.normal;
            }

            scattered = ray(rec.p, scattered_dir);
            attenuation = albedo;
            return true;
        }
};

class metal : public material {
    public:
        color albedo;
        double fuzz;

        metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1){}

        virtual bool scatter (const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        
};

class dielectric : public material {
    public:
        // Index of refraction
        double ir;

        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
            attenuation = color(1.0, 1.0, 1.0);

            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());

            // Account for total internal reflection i.e if refraction isn't possible
            double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            // If no solution to Snell's law (ratio*sin_theta > 1.0 since sin_theta_prime cannot be more than 1.0)
            // Reflect the light, since total internal reflection occuring
            // Otherwise refraction is possible, continue to refract
            bool total_internal_reflection = refraction_ratio * sin_theta > 1.0;

            vec3 direction;
            if (total_internal_reflection || reflectance(cos_theta, refraction_ratio) > random_double()) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, refraction_ratio);
            }

            scattered = ray(rec.p, direction);
            return true;
        }

    private:
        static double reflectance(double cos, double r_idx) {
            // Schlick's appromximation for reflectance
            double r0 = (1 - r_idx) / (1 + r_idx);
            r0 = r0*r0;
            return r0 + (1 - r0)*pow((1 - cos), 5);
        }
};

#endif