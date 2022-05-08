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
#include "load_stb_image.h"
#include "bvh.h"
#include "rect.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define RND_UNIFORM (curand_uniform(&local_rand_state))
#define RND_IN_RANGE(LOW,HIGH) (curand_uniform(&local_rand_state) * (HIGH - LOW) + LOW)
#define PRNT_LOCATION (printf("Frame = %d, x_p = %.6f, y_p = %.6f, z_p = %.6f, x_a %.6f, y_a = %.6f, z_a = %.6f \n", \
frame,(*d_camera)->look_from.x(),(*d_camera)->look_from.y(),(*d_camera)->look_from.z(),(*d_camera)->angles.x(),(*d_camera)->angles.y(),(*d_camera)->angles.z()));

__global__ void create_world(sceneobject **d_list,
                             sceneobject **d_world,
                             camera **d_camera,
                             float aspect_ratio,
                             curandState *rand_state,
                             textureWrap earth_texture,
                             textureWrap mars_texture,
                             textureWrap sunset_texture,
                             textureWrap sky_back_texture,
                             textureWrap sky_bottom_texture,
                             textureWrap sky_left_texture,
                             textureWrap sky_front_texture,
                             textureWrap sky_top_texture,
                             textureWrap sky_right_texture,
                             int num_objects,
                             sceneobject** d_boxes,
                             int num_boxes,
                             sceneobject** d_sky_box,
                             int num_sky_box) {


    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState local_rand_state = *rand_state;

        // Set ground
        auto checker = new checker_texture(color(0.2, 0.3, 0.1), color(0.9,0.9,0.9));
        auto ground_color = new rgb_color(color(0.1, 0.1, 0.1));
        auto light = new diffuse_light(color(7, 7, 7));
        auto earth = new image_texture(earth_texture.width, earth_texture.height, earth_texture.textObj);
        auto mars = new image_texture(mars_texture.width, mars_texture.height, mars_texture.textObj);
        auto sunset = new image_texture(sunset_texture.width, sunset_texture.height, sunset_texture.textObj);


        // neg_x = back, neg_z = left, neg_y = bottom, pos_x = front, pos_y = top, pos_z = right
        auto sky_back = new image_texture(sky_back_texture.width, sky_back_texture.height, sky_back_texture.textObj);
        auto sky_bottom = new image_texture(sky_bottom_texture.width, sky_bottom_texture.height, sky_bottom_texture.textObj);
        auto sky_left = new image_texture(sky_left_texture.width, sky_left_texture.height, sky_left_texture.textObj);
        auto sky_front = new image_texture(sky_front_texture.width, sky_front_texture.height, sky_front_texture.textObj);
        auto sky_top = new image_texture(sky_top_texture.width, sky_top_texture.height, sky_top_texture.textObj);
        auto sky_right = new image_texture(sky_right_texture.width, sky_right_texture.height, sky_right_texture.textObj);

        auto white = new lambertian(color(.73, .73, .73));
        auto red = new lambertian(color(.65, .05, .05));
        auto green = new lambertian(color(.12, .45, .15));

        int i = 0;

        //d_list[i++] = new sphere(vec3(0, -1000.0, 0), 1000.0, new lambertian(ground_color));

        float main_x = RND_IN_RANGE(-4.0, 4.0);
        float main_z = RND_IN_RANGE(-4.0, 4.0);
        float x_buffer = 1.0;
        float z_buffer = 1.0;

        // Big balls
        //d_list[i++] = new sphere(vec3(0,1,0), 1.0, new dielectric(1.5));
        //d_list[i++] = new sphere(vec3(RND_IN_RANGE(-4.0, 4.0),1,RND_IN_RANGE(-4.0, 4.0)), 1.0, new lambertian(earth));
        //d_list[i++] = new sphere(vec3(RND_IN_RANGE(-4.0, 4.0),1,RND_IN_RANGE(-4.0, 4.0)), 1.0, new metal(vec3(0.7,0.6,0.5), 0.0));

        d_list[i++] = new sphere(vec3(200,278,400), 30.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(100,278,350), 30.0, new metal(vec3(0.7,0.6,0.5), 0.0));
        d_list[i++] = new sphere(vec3(250,278,500), 30.0, new lambertian(earth));

        //mars
        d_list[i++] = new sphere(vec3(230,263,300), 15.0, new lambertian(mars));

        //small metal
        d_list[i++] = new sphere(vec3(300,263,400), 15.0, new metal(vec3(0.96,0.25,0.25), 0.15));

        //behind glass
        //d_list[i++] = new sphere(vec3(80,278,500), 30.0, new lambertian(sunset));
        // Ontop of box
        d_list[i++] = new sphere(vec3(100,315,405), 15.0, new lambertian(sunset));

        // bvh cluster
        for (int j = 0; j < num_boxes; j++ ){
            auto x_tmp = RND_IN_RANGE(200,250);
            auto y_tmp = RND_IN_RANGE(310,360);
            auto z_tmp = RND_IN_RANGE(400,450);
            d_boxes[j] = new sphere(point3(x_tmp, y_tmp, z_tmp), 5.0f, white);
        }

        auto boxes = sceneobject_list(d_boxes, num_boxes);
        auto d_bvh = new bvh_node(boxes, rand_state);
        d_list[i++] = d_bvh;

        // Box point3(130, 0, 65), point3(295, 165, 230)
        point3 p0(90, 248, 395);
        point3 p1(110, 300, 415);

        d_list[i++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), green);
        d_list[i++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), red);

        d_list[i++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), green);
        d_list[i++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), red);

        d_list[i++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), green);
        d_list[i++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), red);

        //Subtract large spheres, skybox and light, cube
        int num_small_balls = num_objects - 19;

        for (int nb = 0; nb < num_small_balls; nb++) {
            float choose_material = RND_UNIFORM;
            float sphere_size = RND_IN_RANGE(5.0,10.0);
            float x = RND_IN_RANGE(50,400) + main_x + x_buffer;
            float z = RND_IN_RANGE(250,550) + main_z + z_buffer;

            vec3 center(x, 248 + sphere_size, z);

            if (choose_material < 0.8f) {
                d_list[i++] = new sphere(center, sphere_size,
                                         new lambertian(vec3(RND_UNIFORM*RND_UNIFORM, RND_UNIFORM*RND_UNIFORM, RND_UNIFORM*RND_UNIFORM)));
            } else if (choose_material < 0.95f) {
                d_list[i++] = new sphere(center, sphere_size,
                                         new metal(vec3(0.5f*(1.0f+RND_UNIFORM), 0.5f*(1.0f+RND_UNIFORM), 0.5f*(1.0f+RND_UNIFORM)), 0.5f*RND_UNIFORM));
            } else {
                d_list[i++] = new sphere(center, sphere_size, new dielectric(1.5));
            }
        }

        //Skybox
        float box_dist = 555;

        d_list[i++] = new yz_rect(0, box_dist, 0, box_dist, box_dist, new lambertian_bg(sky_left));
        d_list[i++] = new yz_rect(0, box_dist, 0, box_dist, 0,  new lambertian_bg(sky_right));
        d_list[i++] = new xz_rect(75, 270, 76, 280, 554, light);
        d_list[i++] = new xz_rect(0, box_dist, 0, box_dist, 0, new lambertian_bg(sky_bottom));
        d_list[i++] = new xz_rect(0, box_dist, 0, box_dist, box_dist, new lambertian_bg(sky_top));
        d_list[i++] = new xy_rect(0, box_dist, 0, box_dist, box_dist, new lambertian_bg(sky_front));
        d_list[i++] = new xy_rect(0, box_dist, 0, box_dist, 0, new lambertian_bg(sky_back));

        *rand_state = local_rand_state;
        *d_world = new sceneobject_list(d_list, num_objects);

        // Old Camera
        /*point3 lookfrom(278,278,278);
        point3 lookat(0,278,555);
        float dist_to_focus = 10.0f;
        float aperture = 0.0f;*/

        //point3 lookfrom(350,278,100);
        point3 lookfrom(450,278,200);
        point3 lookat(0,278,555);
        float dist_to_focus = 10.0f;
        float aperture = 0.0f;

        *d_camera = new camera(lookfrom, lookat, vec3(0,0,0), vec3(0,1,0), 40.0, aspect_ratio, aperture, dist_to_focus);
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

__device__ void rotate_scene(camera **d_camera, float new_x_angle, float new_y_angle, float new_z_angle) {
    // Rotations
    if (new_x_angle != (*d_camera)->angles.x()){
        (*d_camera)->rotate_camera_x(new_x_angle);
    }
    if (new_y_angle != (*d_camera)->angles.y()){
        (*d_camera)->rotate_camera_y(new_y_angle);
    }
    if (new_z_angle != (*d_camera)->angles.z()){
        (*d_camera)->rotate_camera_z(new_z_angle);
    }
}


__global__ void update_scene(camera **d_camera, int frame) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        // Scene Example #1
        /* if (frame == 0){
               PRNT_LOCATION
           }
           else if (frame < 180) {
               rotate_scene(d_camera, (*d_camera)->angles.x(), (*d_camera)->angles.y() + 0.25f, (*d_camera)->angles.z());
               if (frame == 179) {
                   PRNT_LOCATION
               }
           }
           else if (frame < 360) {
               rotate_scene(d_camera, (*d_camera)->angles.x(), (*d_camera)->angles.y() + 0.05f, (*d_camera)->angles.z());
               (*d_camera)->translate_camera(-0.05,0.05,0.0);
               if (frame == 359) {
                   PRNT_LOCATION
               }
           }
           else if (frame < 480) {
               rotate_scene(d_camera, (*d_camera)->angles.x(), (*d_camera)->angles.y() + 1.0f, (*d_camera)->angles.z());

               if (frame == 479) {
                   PRNT_LOCATION
               }
           }
           else if (frame < 630) {
               rotate_scene(d_camera, (*d_camera)->angles.x(), (*d_camera)->angles.y() + 0.05f, (*d_camera)->angles.z());
               (*d_camera)->translate_camera(0.0,-0.05,0.05);

               if (frame == 629) {
                   PRNT_LOCATION
               }
           }
           else if (frame < 720) {
               (*d_camera)->translate_camera(0.0,0.00,0.05);

               if (frame == 719) {
                   PRNT_LOCATION
               }
           }*/

        (*d_camera)->compute_camera_scene();

    }
}

__global__ void init_render(int max_x, int max_y, curandState *rand_state, int seed = clock()) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= max_x) || (idx_y >= max_y)) return;

    int pixel_idx = idx_y * max_x + idx_x;

    //rand seed shifted by idx, sequence number, no offset for each thread
    curand_init(seed + pixel_idx, 0, 0, &rand_state[pixel_idx]);
}

__device__ color ray_color(const ray &r, sceneobject **world, curandState *local_rand_state, int ray_bounce_limit, float u, float v) {
    color background = color(255.0, 255.0, 255.0);

    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < ray_bounce_limit; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_attenuation += emitted;
                cur_ray = scattered;
            } else {
                return cur_attenuation * emitted;
            }

        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            //image_texture d_text(sky_texture.width, sky_texture.height, sky_texture.textObj);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation;

            //return cur_attenuation * background;

            //image_texture d_text(sky_texture.width, sky_texture.height, sky_texture.textObj);
            //return cur_attenuation * d_text.value(u, v, vec3(0,0,0));
            //return cur_attenuation * background;
        }
    }
    return cur_attenuation; // exceeded recursion
}

__global__ void render(vec3 *fb, int max_x, int max_y, int num_samples, camera **cam, sceneobject **world, curandState *rand_state,
                       int ray_bounce_limit, textureWrap earth_texture, textureWrap mars_texture, textureWrap sunset_texture) {
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
        col += ray_color(r, world, &local_rand_state, ray_bounce_limit, u, v);
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

textureWrap load_texture(const char* filename) {
    const static uint32_t bytesPerPixel{ 3u };
    uint32_t bytesPerScanline;
    int32_t componentsPerPixel = bytesPerPixel;

    int32_t width, height;
    uint8_t *data;

    data = stbi_load(filename, &width, &height, &componentsPerPixel, componentsPerPixel);

    if(!data) {
        std::cerr << "COULD NOT LOAD IMAGE" << "\n";
        width = height = 0;
    }

    bytesPerScanline = bytesPerPixel * width;

    cudaArray* d_img;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    gpuErrchk(cudaMallocArray(&d_img, &channelDesc, bytesPerScanline, height));

    gpuErrchk(cudaMemcpy2DToArray(d_img, 0, 0, data, bytesPerScanline * sizeof(uint8_t), bytesPerScanline * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

    gpuErrchk(cudaGetLastError());

    cudaTextureObject_t texObj = 0;
    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    cudaTextureDesc textureDesc;
    memset(&textureDesc, 0, sizeof(textureDesc));

    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.pitch2D.width = width;
    resourceDesc.res.pitch2D.height = height;
    resourceDesc.res.array.array = d_img;

    textureDesc.normalizedCoords = true;
    textureDesc.filterMode = cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaAddressModeWrap;
    textureDesc.addressMode[1] = cudaAddressModeWrap;
    textureDesc.readMode = cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&texObj, &resourceDesc, &textureDesc, nullptr));
    gpuErrchk(cudaGetLastError());

    //gpuErrchk(cudaFreeArray(d_img));
    STBI_FREE(data);

    textureWrap text_wrap = {(uint32_t) width, (uint32_t) height, texObj};

    return text_wrap;
}


int main(void) {
    // Image
    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 1920;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int ray_bounce_limit = 25;
    int low_num_samples = 1;
    int high_num_samples = 20;
    int num_boxes = 1000;

    size_t stack_size = 2048;
    cudaThreadSetLimit(cudaLimitStackSize, stack_size); // set stack size

    const int num_pixels = image_width * image_height;

    int num_objects = 50;//12 before small balls;

    int num_images = 2;

    // 32x18 threads works best so far, logic is 32/(16/9) = 18 where 16/9 is our current aspect ratio
    int thread_x = 32;
    int thread_y = 18;

    dim3 blocks(image_width / thread_x + 1, image_height / thread_y + 1);
    dim3 threads(thread_x, thread_y);

    //int seed = 42069;

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


    // Camera
    camera **d_camera;
    gpuErrchk(cudaMalloc((void **) &d_camera, sizeof(camera *)));

    // Build scene
    sceneobject **d_list;
    gpuErrchk(cudaMalloc((void **) &d_list, num_objects * sizeof(sceneobject *)));

    sceneobject **d_world;
    gpuErrchk(cudaMalloc((void **) &d_world, sizeof(sceneobject *)));

    textureWrap earth_texture = load_texture("textures/earthmap.jpg");
    textureWrap mars_texture = load_texture("textures/mars.jpg");
    textureWrap sunset_texture = load_texture("textures/sunset.jpg");

    textureWrap sky_back = load_texture("textures/skybox/back.jpg");
    textureWrap sky_bottom = load_texture("textures/skybox/bottom.jpg");
    textureWrap sky_left = load_texture("textures/skybox/left.jpg");
    textureWrap sky_front = load_texture("textures/skybox/front.jpg");
    textureWrap sky_top = load_texture("textures/skybox/top.jpg");
    textureWrap sky_right = load_texture("textures/skybox/right.jpg");

    // BVH
    sceneobject **d_boxes;
    gpuErrchk(cudaMalloc((void **) &d_boxes, num_boxes * sizeof(sceneobject *)));

    // Skybox
    sceneobject **d_sky_box;
    int num_sky_box = 5;
    gpuErrchk(cudaMalloc((void **) &d_sky_box, num_sky_box * sizeof(sceneobject *)));
/*
    cudaResourceDesc resourceDesc{0};
    memset(&resourceDesc, 0, sizeof(resourceDesc));
    gpuErrchk(cudaGetTextureObjectResourceDesc(&resourceDesc, earth_texture));

    std::cout << "Width recall = " << resourceDesc.res.pitch2D.width << "\n";*/

    create_world<<<1, 1>>>(d_list,
                           d_world,
                           d_camera,
                           aspect_ratio,
                           d_rand_state_world,
                           earth_texture,
                           mars_texture,
                           sunset_texture,
                           sky_back,
                           sky_bottom,
                           sky_left,
                           sky_front,
                           sky_top,
                           sky_right,
                           num_objects,
                           d_boxes,
                           num_boxes,
                           d_sky_box,
                           num_sky_box);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Render
    clock_t start, stop;
    start = clock();

    std::cerr << "Rendering " << num_images << " " << image_width << "x" << image_height << " images ";
    std::cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

    init_render<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());


    size_t fb_size = num_pixels * sizeof(vec3);
    int curr_samples;
    // Generate low-high samples of each image
    for(int i = 0; i < num_images; i++) {

        string image_name="";

        if (i % 2 == 0) {
            std::cerr << "Rendering low image " << i << "/" << num_images << "\n";
            update_scene<<<1,1>>>(d_camera, i);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());

            curr_samples = low_num_samples;
            image_name = "output/ppm_images/image_" + std::to_string(i) + "_low" + ".ppm";
        } else {
            std::cerr << "Rendering high image " << i << "/" << num_images << "\n";
            curr_samples = high_num_samples;
            image_name = "output/ppm_images/image_" + std::to_string(i - 1) + "_high" + ".ppm";
        }

        vec3 *fb;
        gpuErrchk(cudaMallocManaged((void **) &fb, fb_size));

        // Render world
        render<<<blocks, threads>>>(fb, image_width, image_height, curr_samples, d_camera, d_world, d_rand_state,
                                    ray_bounce_limit, earth_texture, mars_texture, sunset_texture);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        output_image(image_width, image_height, fb, image_name);

        // Free old buffer
        gpuErrchk(cudaFree(fb));
    }


    // Generate continuous motion render
/*    for(int i = 0; i < num_images; i++){
        std::cerr << "Rendering image " << i << "/" << num_images << "\n";
        // Init Image buffer
        vec3 *fb;
        gpuErrchk(cudaMallocManaged((void **) &fb, fb_size));

        // Update scene
        update_scene<<<1,1>>>(d_camera, i);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Render world
        render<<<blocks, threads>>>(fb, image_width, image_height, num_samples, d_camera, d_world, d_rand_state,
                                    ray_bounce_limit);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Output image
        string image_name = "output/ppm_images/image_" + std::to_string(i) + ".ppm";
        output_image(image_width, image_height, fb, image_name);

        // Free old buffer
        gpuErrchk(cudaFree(fb));

    }*/

    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Free up
    gpuErrchk(cudaDeviceSynchronize());
    //free_world<<<1, 1>>>(d_list, d_world, d_camera, num_objects);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Destroy texture object
    cudaDestroyTextureObject(sky_back.textObj);
    cudaDestroyTextureObject(sky_bottom.textObj);
    cudaDestroyTextureObject(sky_left.textObj);
    cudaDestroyTextureObject(sky_front.textObj);
    cudaDestroyTextureObject(sky_top.textObj);
    cudaDestroyTextureObject(sky_right.textObj);
    cudaDestroyTextureObject(sunset_texture.textObj);
    cudaDestroyTextureObject(mars_texture.textObj);
    cudaDestroyTextureObject(earth_texture.textObj);

    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaFree(d_sky_box));
    gpuErrchk(cudaFree(d_boxes));
    gpuErrchk(cudaFree(d_world));
    gpuErrchk(cudaFree(d_list));
    gpuErrchk(cudaFree(d_camera));
    gpuErrchk(cudaFree(d_rand_state));
    gpuErrchk(cudaFree(d_rand_state_world));

}