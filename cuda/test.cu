#include <iostream>
#include "load_stb_image.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void test_gpu(cudaTextureObject_t d_img, int32_t width, int32_t height) {
    float u = (threadIdx.x + blockDim.x * blockIdx.x) * 1.0f / width;
    float v = (threadIdx.y + blockDim.y * blockIdx.y) * 1.0f / height;

    v = 1.0f - v;
    uint32_t X = u * width;

    constexpr float pixelOffset{0.5f};
    const float denominator = 1.0f / (float) (3 * width);
    u = (float) (3 * X + pixelOffset);

    float r = (float) tex2D<uint8_t>(d_img, u++ * denominator, v);
    float b = (float) tex2D<uint8_t>(d_img, u++ * denominator, v);
    float g = (float) tex2D<uint8_t>(d_img, u * denominator, v);

    printf("gpu R = %.6f, G = %.6f, B = %.6f \n", r, g, b);

//    }
}

int main(void) {
    std::cout << "hi\n";

    const static uint32_t bytesPerPixel{ 3u };
    uint32_t bytesPerScanline;
    int32_t componentsPerPixel = bytesPerPixel;

    int32_t width, height;
    uint8_t *data;

    const char* filename = "earthmap.jpg";

    data = stbi_load(filename, &width, &height, &componentsPerPixel, componentsPerPixel);

    if(!data) {
        std::cerr << "COULD NOT LOAD IMAGE" << "\n";
        width = height = 0;
    }

    bytesPerScanline = bytesPerPixel * width;

/*    for(int i = 0; i < width * height * componentsPerPixel; i++) {
        std::cout << "r  = " << (float)data[i++] << " g = " << (float)data[i++] << " b = " << (float)data[i] << "\n";
    }*/

    //std::cout << "width = " << width << " height = " << height << " bytes_per_scanline = " << bytesPerScanline << "\n";

    std::cout << "GPU NOW \n";

    cudaArray* d_img;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    gpuErrchk(cudaMallocArray(&d_img, &channelDesc, bytesPerScanline, height));

    gpuErrchk(cudaMemcpy2DToArray(d_img, 0, 0, data, bytesPerScanline * sizeof(uint8_t), bytesPerScanline * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

    //gpuErrchk(cudaMemcpy2DToArray(d_img, 0, 0, data, bytes_per_scanline * sizeof(uint8_t), bytes_per_scanline * sizeof(uint8_t), height, cudaMemcpyHostToDevice));

    gpuErrchk(cudaGetLastError());
    //STBI_FREE(data);
    //gpuErrchk(cudaFreeArray(d_img));

    //gpuErrchk(cudaDeviceSynchronize());

    cudaTextureObject_t textureObject;
    cudaResourceDesc resourceDesc;
    cudaTextureDesc textureDesc;

    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = d_img;

    textureDesc.normalizedCoords = true;
    textureDesc.filterMode = cudaFilterModePoint;
    textureDesc.addressMode[0] = cudaAddressModeWrap;
    textureDesc.addressMode[1] = cudaAddressModeWrap;
    textureDesc.readMode = cudaReadModeElementType;

    gpuErrchk(cudaCreateTextureObject(&textureObject, &resourceDesc, &textureDesc, nullptr));

    int thread_x = 32;
    int thread_y = 18;

    dim3 blocks(width / thread_x + 1, height / thread_y + 1);
    dim3 threads(thread_x, thread_y);

    test_gpu<<<blocks,threads>>>(textureObject, width, height);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFreeArray(d_img));
    STBI_FREE(data);

    std::cout << "end\n";

}