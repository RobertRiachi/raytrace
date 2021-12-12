#ifndef SPHERE_H
#define SPHERE_H

#include "sceneobject.h"

class sphere: public sceneobject {
    public:
        point3 center;
        double radius;
        shared_ptr<material> mat_ptr;

        sphere() {}
        sphere(point3 cen, double r, shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {};

        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    // Given a center and a ray check if the following equation is statisfied: (ray - center)*(ray - center) = radius^2
    // ray = point + t*direction; Hence (point + t*direction - center)*(point + t*direction - center) = radius^2
    // Simplifies to: t^2(direction*direction) + 2t(direction*(point-center)) + (point-center)*(point-center) - r^2 = 0
    // The following solves the quadratic if a solution exists
    vec3 oc = r.origin() - center;

    // vector dot'd with itself is just the squared length
    double a = r.direction().length_squared();
    double b = 2 * dot(oc, r.direction());
    double c = dot(oc, oc) - radius*radius;
    double discriminant = b*b - 4*a*c;

    if (discriminant < 0) return false;

    double root = (-b - sqrt(discriminant))/(2.0*a);

    if (root < t_min || root > t_max) {
        root = root = (-b + sqrt(discriminant))/(2.0*a);

        if (root < t_min || root > t_max) {
            return false;
        }
    }
    
    rec.t = root;
    rec.p = r.at(rec.t);
    // Divide by radius to create unit vector as the magnitude of intersection - center is the radius itself
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}


#endif