#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>
#include "vec3.h"

const double pi = 3.1415926535897932385;

__device__ inline float degrees_to_rads(float degree) { return degree * pi / 180.0f; }

class camera {
public:

    point3 look_from;
    point3 look_at;
    float angle;
    vec3 vup;
    float vfov;
    float aspect_ratio;
    float aperture;
    float focus_dist;

    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 w;
    vec3 u;
    vec3 v;
    float lens_radius;

    __device__ camera(point3 look_from,
                      point3 look_at,
                      float angle,
                      vec3 vup,
                      float vfov,
                      float aspect_ratio,
                      float aperture,
                      float focus_dist): look_from(look_from), look_at(look_at), angle(angle), vup(vup), vfov(vfov),
                                         aspect_ratio(aspect_ratio), aperture(aperture), focus_dist(focus_dist) {

        compute_camera_scene();

        /*
        float theta = degrees_to_rads(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;

        lens_radius = aperture / 2;
         */
    }

    __device__ void compute_camera_scene() {
        float theta = degrees_to_rads(this->vfov);
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = this->aspect_ratio * viewport_height;

        this->w = unit_vector(this->look_from - this->look_at);
        this->u = unit_vector(cross(this->vup, this->w));
        this->v = cross(this->w, this->u);

        this->horizontal = this->focus_dist * viewport_width * this->u;
        this->vertical = this->focus_dist * viewport_height * this->v;
        this->lower_left_corner = this->look_from - this->horizontal / 2 - this->vertical / 2 - this->focus_dist * this->w;

        this->lens_radius = this->aperture / 2;
    }

    // Simple y-axis matrix rotation
    // [cos(theta),  0, sin(theta)] [x]
    // [0,           1,          0] [y]
    // [-sin(theta), 0, cos(theta)] [z]
    __device__ void rotate_camera_y(float new_angle){
        // Compute angle delta
        float d_angle = new_angle - this->angle;
        float d_rads = degrees_to_rads(d_angle);

        float new_x = this->look_from.x()*cos(d_rads) + this->look_from.z()*sin(d_rads);
        float new_y = this->look_from.y();
        float new_z = this->look_from.z()*cos(d_rads) - this->look_from.x()*sin(d_rads);

        this->look_from = point3(new_x, new_y, new_z);
        this->angle = new_angle;

        compute_camera_scene();

    }

    __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {

        vec3 rd = this->lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = this->u * rd.x() + this->v * rd.y();

        return ray(this->look_from + offset, this->lower_left_corner + s * this->horizontal + t * this->vertical - this->look_from - offset);
    }
};

#endif