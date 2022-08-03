# CUDA RayTracing

## Overview
I wanted to explore what developing with CUDA would be like, but simple matrix multiplication wasn't cutting it. I came across Peter Shirley's Ray Tracing series and figured that could be an interesting project. Originally this was written in C++ and then expanded into a cuda implementation which can be found in the cuda subdirectory. Peter's book invites the reader to try and speed up computation as much as possible and thus the motivation for rewritting in cuda.

## Ongoing work
The cuda version is basically complete, I'm currently working on creating a Super Resolution GAN to try and render low res images and quickly infer a higher resolution output somewhat similar to Nvidia's DLSS.

## How to run

### CPU
Simply compile and run the C++ code in the main project directory. This will output renders in PPM format.

### CUDA
cd into the cuda subdirectory, with cuda v11.0 or higher, and set the arch flag to your GPU's architecture. I chain the command to a conversion script for simplicity.
```
nvcc -arch=sm_75 -o a.out main.cu && ./a.out && python3 output/convert.py
```

### Converting PPM to PNG
Personally I use imagemagick, the convert.py script can be invoked to automatically do this for you though.
```
sudo apt install imagemagick
convert image_new.ppm image_new.png
```

### Outputing Video
Install Requirements
```
 sudo apt install ffmpeg
```
After rendering multiple frames in sequence, run the generate_video script:
```
python3 output/generate_video.py
```
