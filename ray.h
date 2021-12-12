#ifndef RAY_H
#define RAY_H

#include "vec3.h"

// Define a ray class of the form Point(t) = origin + t*direction
// Computes point t along the ray defined by origin, direction
class ray {
    public:
        point3 orig;
        vec3 dir;

        ray() {}
        ray(const point3& origin, const vec3& direction): orig(origin), dir(direction) {}

        point3 origin() const {return orig;}
        vec3 direction() const {return dir;}

        point3 at(double t) const {
            return orig + t*dir;
        }
};


#endif