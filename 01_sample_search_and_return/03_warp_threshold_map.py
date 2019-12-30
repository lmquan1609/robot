import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import *

def rover_coords(thresholded_img):
    y0s, x0s = thresholded_image.nonzero()

    y_pixel = -(x0s - thresholded_img.shape[1]/2).astype('float')
    x_pixel = -(y0s - thresholded_img.shape[0]).astype('float')

    return x_pixel, y_pixel

image = mpimg.imread('sample.jpg')

warped = perspect_transform(image, source, destination)
thresholded_image = color_thresh(warped, lower_rgb_thresh=(160, 160, 160))

xpix, ypix = rover_coords(thresholded_image)


plt.subplot(121)
ypos, xpos = thresholded_image.nonzero()
plt.plot(xpos, ypos, '.')

# Plot the map in rover-centric coords
fig = plt.figure(figsize=(5, 7.5))

plt.subplot(122)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
plt.title('Rover-Centric Map', fontsize=20)
plt.show()  # Uncomment if running on your local machine