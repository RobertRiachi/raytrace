#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <stdlib.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "sceneobject_list.h"
#include "camera.h"
#include "material.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define RND_UNIFORM (curand_uniform(&local_rand_state))

__global__ void create_world(sceneobject **d_list, sceneobject **d_world, camera **d_camera, float aspect_ratio, curandState *rand_state, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;

        // Set ground
        d_list[0] = new sphere(vec3(0, -1000.0, 0), 1000.0, new lambertian(vec3(0.5, 0.5, 0.5)));

        int i = 1;

        // Big balls
        d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4,1,0), 1.0, new lambertian(vec3(0.4,0.2,0.1)));
        d_list[i++] = new sphere(vec3(4,1,0), 1.0, new metal(vec3(0.7,0.6,0.5), 0.0));

        // Remaining balls
        for (int a = -11; a < 11; a++ ) {
            for (int b = -11; b < 11; b++) {

                float choose_material = RND_UNIFORM;
                vec3 center(a + RND_UNIFORM, 0.2, b + RND_UNIFORM);

                if (choose_material < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND_UNIFORM*RND_UNIFORM, RND_UNIFORM*RND_UNIFORM, RND_UNIFORM*RND_UNIFORM)));
                } else if (choose_material < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND_UNIFORM), 0.5f*(1.0f+RND_UNIFORM), 0.5f*(1.0f+RND_UNIFORM)), 0.5f*RND_UNIFORM));
                } else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }

            }
        }

        *rand_state = local_rand_state;
        *d_world = new sceneobject_list(d_list, num_objects);

        //Camera
        point3 lookfrom(13,2,5);
        point3 lookat(0,0,0);
        float dist_to_focus = 10.0f;
        float aperture = 0.1f;

        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, aspect_ratio, aperture, dist_to_focus);
    }
}

__global__ void free_world(sceneobject **d_list, sceneobject **d_world, camera **d_camera, int num_objects) {
    for(int i=0; i < num_objects; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void init_world(curandState *rand_state, int seed = clock()) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(seed, 0, 0, rand_state);
    }
}

/*
__global__ void init_objects(curandState *rand_state, int num_objects, int seed) {
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (t_idx >= num_objects) return;

    curand_init(seed)
}
*/

__global__ void init_render(int max_x, int max_y, curandState *rand_state, int seed = clock()) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= max_x) || (idx_y >= max_y)) return;

    int pixel_idx = idx_y * max_x + idx_x;

    //rand seed shifted by idx, sequence number, no offset for each thread
    curand_init(seed + pixel_idx, 0, 0, &rand_state[pixel_idx]);
}

__device__ color ray_color(const ray &r, sceneobject **world, curandState *local_rand_state, int ray_bounce_limit) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < ray_bounce_limit; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return color(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return color(0.0,0.0,0.0); // exceeded recursion
}

__global__ void render(vec3 *fb, int max_x, int max_y, int num_samples, camera **cam, sceneobject **world, curandState *rand_state,
       int ray_bounce_limit) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // If out of bounds stop
    if ((idx_x >= max_x) || (idx_y >= max_y)) return;

    int pixel_idx = idx_y * max_x + idx_x;

    curandState local_rand_state = rand_state[pixel_idx];
    color col(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        float u = float(idx_x + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(idx_y + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state, ray_bounce_limit);
    }
    rand_state[pixel_idx] = local_rand_state;
    col /= float(num_samples);
    col.e[0] = sqrt(col.e[0]);
    col.e[1] = sqrt(col.e[1]);
    col.e[2] = sqrt(col.e[2]);
    fb[pixel_idx] = col;

}


void output_image(int image_width, int image_height, vec3 *fb, string image_name) {

    std::ofstream ofs(image_name, std::ofstream::trunc);

    ofs << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_idx = j * image_width + i;

            vec3 pixel = fb[pixel_idx];

            float r = pixel.x();
            float g = pixel.y();
            float b = pixel.z();

            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);

            ofs << ir << " " << ig << " " << ib << "\n";
        }
    }

    ofs.close();
}

int main(void) {
    // Image
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    //const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int num_samples = 10;
    const int ray_bounce_limit = 50;

    const int num_pixels = image_width * image_height;

    int num_objects = 22*22+1+3;
    //int seed = 42069;

    size_t fb_size = num_pixels * sizeof(vec3);

    // Image buffer
    vec3 *fb;
    gpuErrchk(cudaMallocManaged((void **) &fb, fb_size));

    // Init rand state
    curandState *d_rand_state;
    gpuErrchk(cudaMalloc((void **) &d_rand_state, num_pixels * sizeof(curandState)));

    curandState *d_rand_state_world;
    gpuErrchk(cudaMalloc((void **)&d_rand_state_world, 1*sizeof(curandState)));

    curandState *d_rand_state_objects;
    gpuErrchk(cudaMalloc((void **) &d_rand_state_objects, num_objects*sizeof(curandState)));

    init_world<<<1,1>>>(d_rand_state_world);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    //init_objects<<<1,1>>>(d_rand_state_objects, seed);


    // Camera
    camera **d_camera;
    gpuErrchk(cudaMalloc((void **) &d_camera, sizeof(camera *)));

    // Build scene
    sceneobject **d_list;
    gpuErrchk(cudaMalloc((void **) &d_list, num_objects * sizeof(sceneobject *)));

    sceneobject **d_world;
    gpuErrchk(cudaMalloc((void **) &d_world, sizeof(sceneobject *)));

    create_world<<<1, 1>>>(d_list, d_world, d_camera, aspect_ratio, d_rand_state_world, num_objects);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());


    // Render
    // 32x18 threads works best so far, logic is 32/(16/9) = 18 where 16/9 is our current aspect ratio
    int thread_x = 32;
    int thread_y = 18;

    clock_t start, stop;
    start = clock();

    dim3 blocks(image_width / thread_x + 1, image_height / thread_y + 1);
    dim3 threads(thread_x, thread_y);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

    init_render<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, image_width, image_height, num_samples, d_camera, d_world, d_rand_state,
                                ray_bounce_limit);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output image
    output_image(image_width, image_height, fb, "image_new.ppm");

    // Free up
    gpuErrchk(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera, num_objects);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_world));
    gpuErrchk(cudaFree(d_list));
    gpuErrchk(cudaFree(d_camera));
    gpuErrchk(cudaFree(d_rand_state));
    gpuErrchk(cudaFree(d_rand_state_world));
    gpuErrchk(cudaFree(fb));

}