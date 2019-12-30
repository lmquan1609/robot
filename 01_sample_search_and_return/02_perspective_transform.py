import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    
    return warped

image = mpimg.imread('images/example_grid1.jpg')
plt.subplot(121)
plt.imshow(image)

source = np.float32([[14.8 ,140.7 ], [ 300.9, 140.7 ], [ 118.5,96.5 ], [ 201.2,96.5 ]])

destination = np.float32([[ (image.shape[1]/2)-5, 150 ], [ (image.shape[1]/2)+5,150 ], [(image.shape[1]/2)-5 , 140 ], [ (image.shape[1]/2)+5,140 ]])      

warped = perspect_transform(image, source, destination)
plt.subplot(122)
plt.imshow(warped)
plt.show() 