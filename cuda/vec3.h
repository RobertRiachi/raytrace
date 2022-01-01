#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

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

        __host__ __device__ vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }
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

// convert to unit vector
__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v/v.length();
}

#endif