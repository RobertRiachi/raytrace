#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

#include "utils.h"

class vec3 {
    public:

        float e[3];

        __host__ __device__ vec3(): e{0,0,0} {}
        __host__ __device__ vec3(float p1, float p2, float p3): e{p1,p2,p3} {}
        __host__ __device__ inline float x() const {return e[0];}
        __host__ __device__ inline float y() const {return e[1];}
        __host__ __device__ inline float z() const {return e[2];}

        __host__ __device__ float length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }
        __host__ __device__ float length() const {return sqrt(length_squared());}

        __host__ __device__ float operator[](int i) const {return e[i]; }
        __host__ __device__ float& operator[](int i) {return e[i]; }

        __host__ __device__ vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }
        __host__ __device__ vec3& operator*=(const vec3 &v) {
            e[0] *= v.e[0];
            e[1] *= v.e[1];
            e[2] *= v.e[2];
            return *this;
        }

        __host__ __device__ inline vec3& operator/=(const float t);
        __host__ __device__ inline vec3 operator-() const {return vec3(-e[0], -e[1], -e[2]); }

};

// Aliases
using point3 = vec3; // 3D point
using color = vec3; // RGB color

// add two vectors
__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

// subtract two vectors
__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// multiply two vectors
__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

// multiply vector with a scalar
__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t* v.e[0], t* v.e[1], t* v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)  {
    return t * v;
}

// vector division & invert
__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

// dot product of two vectors
__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

// cross product of two vectors
__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    float i = u.e[1] * v.e[2] - u.e[2] * v.e[1];
    float j = u.e[2] * v.e[0] - u.e[0] * v.e[2];
    float k = u.e[0] * v.e[1] - u.e[1] * v.e[0];

    return vec3(i,j,k);
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

// convert to unit vector
__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v/v.length();
}

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 rand_vec = vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
    auto p = 2.0f * rand_vec - vec3(1, 1, 1);

    while (dot(p,p) >= 1.0f) {
        // Keep generating because it's outside of the unit sphere
        rand_vec = vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
        p = 2.0f * rand_vec - vec3(1, 1, 1);
    }
    return p;
}

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {

    // Generates random floats between -1 and 1 for x and y
    vec3 p = 2.0f*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1,1,0);
    while (dot(p,p) >= 1.0f) {
        // Keep generating because it's outside of unit disk
        p = 2.0f*vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1,1,0);
    }
    return p;
/*    auto r = curand_uniform(local_rand_state);
    auto th = curand_uniform(local_rand_state) * 2 * pi;
    return vec3(r * cos(th), r * sin(th), 0);*/
}

__device__ vec3 random_in_hemisphere(const vec3 &normal, curandState *local_rand_state) {
    vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
    // In same hemisphere as normal
    if (dot(in_unit_sphere, normal) > 0.0f) {
        return in_unit_sphere;
    } else {
        return -in_unit_sphere;
    }
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f*dot(v,n)*n;
}

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif