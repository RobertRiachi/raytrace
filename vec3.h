#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include "utils.h"

using std::sqrt;
using std::fabs;
using std::fmin;

class vec3 {
    public:
        // 3D vector
        double e[3];

        // Constructors
        vec3() : e{0,0,0} {}
        vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        inline double x() const {return e[0];}
        inline double y() const {return e[1];}
        inline double z() const {return e[2];}
        inline double r() const {return e[0];}
        inline double g() const {return e[1];}
        inline double b() const {return e[2];}

        // Operations
        vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]); }
        inline double operator[](int i) const {return e[i]; }
        inline double& operator[](int i) {return e[i]; }


        vec3& operator*=(const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        vec3& operator/=(const double t) {return *this *= 1/t;}

        double length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        double length() const {return sqrt(length_squared());}

        inline static vec3 random() {
            return vec3(random_double(), random_double(), random_double());
        }
        inline static vec3 random(double min, double max) {
            return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
        }

        bool near_zero() const {
            const float threshold = 1e-8;
            // Return true if all dimensions of vector smaller than some threshold
            return (fabs(e[0]) < threshold) && (fabs(e[1]) < threshold) && (fabs(e[2]) < threshold);
        }
};

// Aliases
using point3 = vec3; // 3D point
using color = vec3; // RGB color

// vec3 helpers

// print vector
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

// add two vectors
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

// subtract two vectors
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// multiply two vectors
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

// multiply vector with a scalar
inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t* v.e[0], t* v.e[1], t* v.e[2]);
}

inline vec3 operator*(const vec3 &v, double t)  {
    return t * v;
}

// vector division & invert
inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

// dot product of two vectors
inline double dot(const vec3 &u, const vec3 &v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

// cross product of two vectors
inline vec3 cross(const vec3 &u, const vec3 &v) {
    double i = u[1] * v[2] - u[2] * v[1];
    double j = u[2] * v[0] - u[0] * v[2];
    double k = u[0] * v[1] - u[1] * v[0];

    return vec3(i,j,k);
}

// convert to unit vector
inline vec3 unit_vector(vec3 v) {
    return v/v.length();
}

vec3 random_in_unit_sphere() {
    vec3 p = vec3::random(-1,1);
    while (p.length_squared() >= 1) {
        p = vec3::random(-1,1);
    }
    return p;
}

// Pick points on the surface of the unit sphere, by picking a point in the sphere and normalizing it
vec3 random_unit_vector(){
    return unit_vector(random_in_unit_sphere());
}

vec3 random_in_hemisphere(const vec3& normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();

    // In the same hemisphere as the normal vector
    return  dot(in_unit_sphere, normal) > 0.0 ? in_unit_sphere : -in_unit_sphere;
}

vec3 random_in_unit_disk() {
    // Generate random point from disk centered at radius zero
    vec3 p = vec3(random_double(-1,1), random_double(-1,1), 0);
    while(p.length_squared() >= 1){
        p = vec3(random_double(-1,1), random_double(-1,1), 0);
    }
    return p;
}

vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    //
    double cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif