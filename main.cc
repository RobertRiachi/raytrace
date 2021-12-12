#include <iostream>

#include "utils.h"
#include "color.h"
#include "sceneobject_list.h"
#include "sphere.h"
#include "camera.h"
#include "material.h"

color ray_color(const ray& r, const sceneobject_list& world, int bounce_limit) {

    // If we've exceeded ray bounce limit, no more light gathered
    if (bounce_limit <= 0)
        return color(0,0,0);

    // Check if we hit an object in the scene, if so compute color with normal vec of hit intersection
    hit_record rec;
    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;

        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return attenuation * ray_color(scattered, world, bounce_limit - 1);
        }

        return color(0,0,0);
    }

    // Convert to unit vector
    vec3 unit_direction = unit_vector(r.direction());
    // Normalize between [-1,1] to create gradient in the y-axis
    double t = 0.5*(unit_direction.y() + 1.0);
    // Blend colors of choice by adding them as complements relative to the position of the ray
    return (1.0 - t)*color(1.0,1.0,1.0) + t*color(0.5, 0.7, 1.0); 
}

sceneobject_list generate_random_scene() {
    sceneobject_list world;

    // Build ground
    shared_ptr<lambertian> ground_material = make_shared<lambertian>(color(0.5,0.5,0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_material = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4,0.2,0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                // diffuse material
                if (choose_material < 0.8) {
                    color albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);

                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                // reflective material (metal)
                else if (choose_material < 0.95) {
                    color albedo = color::random(0.5, 1);
                    double fuzz = random_double(0,0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                // reflect and refract material (glass)
                else {
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    shared_ptr<dielectric> material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(-1,1,3), 1.0, material1));

    shared_ptr<lambertian> material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-2.5, 1, -1), 1.0, material2));

    shared_ptr<metal> material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(2.5, 1, 0), 1.0, material3));

    return world;
}

int main() {

    // Define Image Constants
    const float aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width/aspect_ratio);
    const int samples_per_pixel = 500;
    const int ray_bounce_limit = 50;

    // Build Objects in Scene
    sceneobject_list world = generate_random_scene();

    // Camera
    point3 lookfrom(6,3, -13);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Output render
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {

            color pixel_color(0,0,0);
            for (int s = 0; s < samples_per_pixel; s++) {
                double u = (i + random_double()) / (image_width-1);
                double v = (j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u,v);

                pixel_color += ray_color(r, world, ray_bounce_limit);

            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}