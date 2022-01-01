#include <iostream>
#include <math.h>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "sceneobject_list.h"
#include "camera.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void create_world(sceneobject **d_list, sceneobject **d_world, camera **d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world = new sceneobject_list(d_list,2);
    *d_camera = new camera();
  }
}

__global__ void free_world(sceneobject **d_list, sceneobject **d_world, camera **d_camera) {
   delete *(d_list);
   delete *(d_list+1);
   delete *d_world;
   delete *d_camera;
}

__global__ void init(int max_x, int max_y, curandState *rand_state) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((idx_x >= max_x) || (idx_y >= max_y)) return;

  int pixel_idx = idx_y*max_x + idx_x;

  //69420 rand seed, diff sequence number, no offset for each thread
  curand_init(69420, pixel_idx, 0, &rand_state[pixel_idx]);
}

__device__ color ray_color(const ray& r, sceneobject **world) {
  hit_record rec;

  if((*world)->hit(r, 0.0, FLT_MAX, rec)) {
    return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
  }

  vec3 unit_dir = unit_vector(r.direction());
  float t = 0.5f*(unit_dir.y() + 1.0f);
  return (1.0f - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int num_samples, camera **cam, sceneobject **world, curandState *rand_state) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  // If out of bounds stop
  if((idx_x >= max_x) || (idx_y >= max_y)) return;

  int pixel_idx = idx_y * max_x + idx_x;

  curandState local_rand_state = rand_state[pixel_idx];
  color col(0,0,0);
  for (int s = 0; s < num_samples; s++) {
    float u = float(idx_x + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(idx_y + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u,v);
    col += ray_color(r, world);
  }

  fb[pixel_idx] = col/float(num_samples);

}

int main(void)
{
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 400;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int num_samples = 100;

  const int num_pixels = image_width * image_height;

  size_t fb_size = num_pixels * sizeof(vec3);

  // Image buffer
  vec3 *fb;
  gpuErrchk(cudaMallocManaged((void **)&fb, fb_size));

  // Init rand state
  curandState *d_rand_state;
  gpuErrchk(cudaMalloc((void **)& d_rand_state, num_pixels*sizeof(curandState)));

  // Camera
  camera **d_camera;
  gpuErrchk(cudaMalloc((void **)&d_camera, 2*sizeof(camera *)));

  // Build scene
  sceneobject **d_list;
  gpuErrchk(cudaMalloc((void **)& d_list, 2*sizeof(sceneobject *)));

  sceneobject **d_world;
  gpuErrchk(cudaMalloc((void **)& d_world, sizeof(sceneobject *)));

  create_world<<<1,1>>>(d_list, d_world, d_camera);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());


  // Render 
  int thread_x = 32;
  int thread_y = 32;

  clock_t start, stop;
  start = clock();

  dim3 blocks(image_width/thread_x + 1, image_height/thread_y + 1);
  dim3 threads(thread_x, thread_y);

  std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
  std::cerr << "in " << thread_x << "x" << thread_y << " blocks.\n";

  init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(fb, image_width, image_height, num_samples, d_camera, d_world, d_rand_state);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output image

  std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

  for (int j = image_height-1; j >=0; j--) {
    for (int i = 0; i < image_width; i++) {
      size_t pixel_idx = j*image_width + i;

      vec3 pixel = fb[pixel_idx];

      float r = pixel.x();
      float g = pixel.y();
      float b = pixel.z();

      int ir = int(255.99 * r);
      int ig = int(255.99 * g);
      int ib = int(255.99 * b);

      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // Free up
  gpuErrchk(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list, d_world, d_camera);

  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaFree(d_world));
  gpuErrchk(cudaFree(d_list));
  gpuErrchk(cudaFree(d_camera));
  gpuErrchk(cudaFree(d_rand_state));
  gpuErrchk(cudaFree(fb));
 
}