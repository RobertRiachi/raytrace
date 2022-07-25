# Dummy script to convert all ppm files to png
import os

INPUT_FOLDER =  "/ppm_images"
LOW_OUTPUT_FOLDER = "/png_images/low"
HIGH_OUTPUT_FOLDER = "/png_images/high"

directory = os.path.dirname(os.path.realpath(__file__))

num_converted = 0
for filename in os.listdir(directory  + INPUT_FOLDER):

    if filename.endswith(".ppm"):

        new_filename = ("_").join(filename.split(".")[0].split("_")[:-1]) + ".png"

        resolution = filename.split('_')[-1].split('.')[0]

        f_1 = directory + INPUT_FOLDER + "/" + filename

        if resolution == "low":
            f_2 = directory + LOW_OUTPUT_FOLDER + "/" + new_filename
        else:
            f_2 = directory + HIGH_OUTPUT_FOLDER + "/" + new_filename
        
        #elif resolution == "high":
        #    f_2 = directory + HIGH_OUTPUT_FOLDER + "/" + new_filename

        os.system("convert " + f_1 + " " + f_2)
        num_converted += 1

print(f"Converted {num_converted} files")
