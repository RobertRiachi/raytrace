#ifndef TEXTURE_H
#define TEXTURE_H

#include "vec3.h"

struct textureWrap {
    uint32_t width;
    uint32_t height;
    cudaTextureObject_t textObj;
};

class custom_texture {
    public:
        __device__ virtual color value(float u, float v, const vec3& p) const = 0;
};

class rgb_color: public custom_texture {
    public:
        color color_value;

        __device__ rgb_color() {}
        __device__ rgb_color(color c) : color_value(c) {}

        __device__ rgb_color(float red, float green, float blue) : rgb_color(color(red,green,blue)) {}

        __device__ color value(float u, float v, const vec3& p) const override {
            return color_value;
        }

};

class checker_texture: public custom_texture {
    public:
        custom_texture *even;
        custom_texture *odd;

        __device__ checker_texture() {}

        __device__ checker_texture(custom_texture* even, custom_texture* odd) : even(even), odd(odd) {}

        __device__ checker_texture(color c1, color c2) : even(new rgb_color(c1)), odd(new rgb_color(c2)) {}

        __device__ color value(float u, float v, const vec3& p) const override {
            float sines = sin(10.0f*p.x()) * sin(10.0f*p.y()) * sin(10.0f*p.z());

            if (sines < 0.0f) {
                return odd->value(u, v, p);
            } else {
                return even->value(u, v, p);
            }

        }
};

class image_texture: public custom_texture {
    public:
        cudaTextureObject_t d_texture;
        uint32_t texture_width;
        uint32_t texture_height;

        __device__ image_texture(uint32_t width, uint32_t height, cudaTextureObject_t textureObj) : texture_width(width), texture_height(height), d_texture(textureObj) {}

        __device__ color value(float u, float v, const vec3& p) const override {
            v = 1.0f - v;
            uint32_t position = u * texture_width;
            float offset = 0.5f;
            float denominator = 1.0f / (float)(3 * texture_width);
            u = (float)(3.0f * position + offset);

            float r = (float) tex2D<uint8_t>(d_texture, u++ * denominator, v) / 255.0f;
            float b = (float) tex2D<uint8_t>(d_texture, u++ * denominator, v) / 255.0f;
            float g = (float) tex2D<uint8_t>(d_texture, u * denominator, v) / 255.0f;

            return color(r,b,g);

        }

};

#endif
