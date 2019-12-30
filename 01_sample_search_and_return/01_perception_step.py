import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def mask_at_threshold(np_array, threshold):
    return np.where(np_array < threshold, 0, 1)

def color_thresh(img, lower_rgb_thresh=(0, 0, 0), upper_rgb_thresh=(255, 255, 255)):
    output = np.zeros_like(img[:, :, 0])

    mask = (img[:, :, 0] > upper_rgb_thresh[0]) & (img[:, :, 1] > upper_rgb_thresh[1]) & (img[:, :, 2] > upper_rgb_thresh[2])

    output[mask] = 1

    mask = (img[:, :, 0] < lower_rgb_thresh[0]) & (img[:, :, 1] < lower_rgb_thresh[1]) & (img[:, :, 2] < lower_rgb_thresh[2])

    output[mask] = 1
    output[~mask] = 0
    
    return output

filename = 'sample.jpg'

image = mpimg.imread(filename)

lower_threshold = (160, 160, 160)

thresholded_image = color_thresh(image, lower_rgb_thresh=lower_threshold)

# Display the original image and binary               
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(thresholded_image, cmap='gray')
ax2.set_title('thresholded', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


plt.show()