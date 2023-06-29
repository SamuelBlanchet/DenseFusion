import os
import numpy as np
from PIL import Image
from numpy import asarray
from numpy import savetxt
import scipy

exo_image_path = 'distances.png'
linemod_image_path = 'linemod_depth.png'

if __name__ == '__main__':


    depth = np.array(Image.open('linemod_depth.png'))
    np.savetxt("linemod_pixels.txt", depth.astype(int), fmt='%i')
    
    gray = (depth).astype('uint8')
    img = Image.fromarray(depth)

    # Save the grayscale image as a PNG file
    # print the shape of the array
    print("Shape:", depth.shape)

    # print the data type of the array
    print("Data type:", depth.dtype)

    # print the number of dimensions of the array
    print("Number of dimensions:", depth.ndim)

    # print the total number of elements in the array
    print("Number of elements:", depth.size)

    # print the format of the values contained in the array
    print("Format of values:", depth.dtype.name)

    # print the memory address of the first element in the array
    print("Memory address of first element:", depth.data)
    img.save('depth_linemod.png')

    # Exo
    file_name = "1873.ppm"
    with open(file_name, "r") as ppm_file:
        format = ppm_file.readline().split()
        size = ppm_file.readline().split()
        width = size[0]
        height = size[1]
        max_val = int(ppm_file.readline().split()[0])
        # Read the PPM pixel data
        pixels = []
        for line in ppm_file:
            # Split each line into integers
            row = [int(x) for x in line.split()]
            pixels.append(row)
    # Convert the list of rows to a 2D numpy array
    pixels = np.array(pixels).astype(np.int32)
    depth_reshaped = np.reshape(pixels, (480, 640, 3))
    pixels = np.mean(depth_reshaped, axis=2)
    pixels = pixels.astype(np.int32)
    np.savetxt("my_pixels.txt", pixels.astype(int), fmt='%i')
    
    # print the shape of the array
    print("Shape:", pixels.shape)

    # print the data type of the array
    print("Data type:", pixels.dtype)

    # print the number of dimensions of the array
    print("Number of dimensions:", pixels.ndim)

    # print the total number of elements in the array
    print("Number of elements:", pixels.size)

    # print the format of the values contained in the array
    print("Format of values:", pixels.dtype.name)

    # print the memory address of the first element in the array
    print("Memory address of first element:", pixels.data)

    gray = (pixels).astype('uint32')     # int32
    img = Image.fromarray(pixels)
    # Save the grayscale image as a PNG file
    img.save('pixels.png')
