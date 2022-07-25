import os
fps = 60
current_dir = os.path.dirname(os.path.realpath(__file__))
resolution = 'low'
resolution_dir = "/png_images/" + resolution + "/"
input = current_dir + resolution_dir + "image_%01d.png"
output = current_dir + "/render_video.mp4"

# For best output set crf = 0 for lossless
crf = 20

os.system(f"ffmpeg -r {fps} -i {input} -c:v libx264 -crf {crf} {output}")