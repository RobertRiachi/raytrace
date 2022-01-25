#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>
#include "vec3.h"

const double pi = 3.1415926535897932385;
__device__ inline float degrees_to_rads(float degree) {return degree * pi / 180.0f;}

class camera {
    public:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 w;
        vec3 u;
        vec3 v;
        float lens_radius;

        __device__ camera(point3 lookfrom,
                          point3 lookat,
                          vec3 vup,
                          float vfov,
                          float aspect_ratio,
                          float aperture,
                          float focus_dist) {

            float theta = degrees_to_rads(vfov);
            float h = tan(theta/2);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2  - focus_dist*w;

            lens_radius = aperture / 2;
        }


        //__host__ __device__ void camera_update_lookfrom(point3 lookfrom)

        __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {

            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }
};

#endif