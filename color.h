#ifndef COLOR_H
#define COLOR_H


#include "vec3.h"

#include <iostream>

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) {

    double r = pixel_color.r();
    double g = pixel_color.g();
    double b = pixel_color.b();

    // Divide color by samples and gamma-correct with gamme=2.0

    float scale = 1.0/samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Output each value of the color vec3 vector as a value between [0,255]
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

#endif