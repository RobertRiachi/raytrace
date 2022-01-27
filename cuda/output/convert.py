# Dummy script to convert all ppm files to png
import os

INPUT_FOLDER =  "/ppm_images"
OUTPUT_FOLDER = "/png_images"

directory = os.path.dirname(os.path.realpath(__file__))

num_converted = 0
for filename in os.listdir(directory  + INPUT_FOLDER):

    if filename.endswith(".ppm"):

        new_filename = (".").join(filename.split(".")[:-1]) + ".png"

        f_1 = directory + INPUT_FOLDER + "/" + filename
        f_2 = directory + OUTPUT_FOLDER + "/" + new_filename

        os.system("convert " + f_1 + " " + f_2)
        num_converted += 1

print(f"Converted {num_converted} files")
