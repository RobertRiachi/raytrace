import os
fps = 60
current_dir = os.path.dirname(os.path.realpath(__file__))
input = current_dir + "/png_images/image_%01d.png"
output = current_dir + "/1920x1080-60fps.mp4"

# For best output set crf = 0 for lossless
crf = 20

os.system(f"ffmpeg -r {fps} -i {input} -c:v libx264 -crf {crf} {output}")