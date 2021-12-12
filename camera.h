#ifndef CAMERA_H
#define CAMERA_H

#include "utils.h"

class camera{
    public:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 focal_length;
        vec3 u;
        vec3 v;
        vec3 w;
        double lens_radius;

        //vertival field of view in degrees, aspect_ratio
        camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, 
                double aspect_ratio, double aperature, double focus_dist) {
            
            // Define Scene Camera centered at lookfrom
            double theta = degrees_to_rads(vfov);
            double h = tan(theta/2);
            double viewport_height = 2.0 * h;
            double viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperature / 2;
        }

        ray get_ray(double s, double t) const {
            vec3 rd = lens_radius * random_in_unit_disk();
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(origin + offset,lower_left_corner + s*horizontal + t*vertical - origin - offset);
        }

};

#endif