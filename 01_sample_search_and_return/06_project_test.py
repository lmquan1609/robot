import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc # For saving images as needed
import glob # For reading in a list of images
import cv2
import pandas as pd
from collections import namedtuple
from utils import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip

# Data
image_list = glob.glob('test_dataset/IMG/*')

# Read in saved data and ground truth map of the world
df = pd.read_csv('test_dataset/robot_log.csv', delimiter=';', decimal='.')
csv_image_list = df['Path'].tolist()
print(csv_image_list[0])

# Read the ground truth map and stack its depth to create 3D
ground_truth = mpimg.imread('calibration_images/map_bw.png')
ground_truth_3d = np.dstack([ground_truth * 0, ground_truth * 255, ground_truth * 0]).astype('float')

# Data container
# Creating a class to be the data container
# Will read in saved data from csv file and populate this object
# Worldmap is instantiated as 200 x 200 grids corresponding 
# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)
# This encompasses the full range of output position values in x and y from the sim
class Databucket():
    def __init__(self):
        self.images = csv_image_list  
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = 0 # This will be a running index
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap

data = Databucket()

def process_image(img):
    # First create a blank image
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1] * 2, 3))
    output_image[:img.shape[0], :img.shape[1]] = img

    # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
    output_image[:img.shape[0], img.shape[1]:] = warped

    # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    output_image[img.shape[0]:, :data.worldmap.shape[1]] = np.flipud(map_add)

    # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image

# Define pathname to save the output video
output = 'test_mapping.mp4'
data = Databucket() # Re-initialize data in case you're running this cell multiple times
clip = ImageSequenceClip(data.images, fps=60) # Note: output video will be sped up because 
                                          # recording rate in simulator is fps=25
new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
new_clip.write_videofile(output, audio=False)