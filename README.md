# CUDA RayTracing

# Converting PPM to PNG
Currently outputing .ppm files as it's very simple with cpp to easily convert to png:
```
sudo apt install imagemagick
convert image_new.ppm image_new.png
```

# Compiling & Running
To compile use Nvidia's Cuda compiler, then chain with conversion script:
```
nvcc -o a.out main.cu && ./a.out && python3 output/convert.py
```

# Outputing Video
Install Requirements
```
 sudo apt install ffmpeg
```
Run the generate_video Python script after installing the requirements:
```
python3 output/generate_video.py
```

# TODO
- Finish raytracing code
- Take in custom input
- Rotate camera around scene
- Optimize CUDA code
- Output directly to png