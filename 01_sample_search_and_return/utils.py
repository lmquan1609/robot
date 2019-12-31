import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def color_thresh(img, lower_rgb_thresh=(0, 0, 0), upper_rgb_thresh=(255, 255, 255)):
    output = np.zeros_like(img[:, :, 0])

    mask = (img[:, :, 0] > upper_rgb_thresh[0]) & (img[:, :, 1] > upper_rgb_thresh[1]) & (img[:, :, 2] > upper_rgb_thresh[2])

    output[mask] = 1

    mask = (img[:, :, 0] < lower_rgb_thresh[0]) & (img[:, :, 1] < lower_rgb_thresh[1]) & (img[:, :, 2] < lower_rgb_thresh[2])

    output[mask] = 1
    output[~mask] = 0
    
    return output

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    # Apply a rotation
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos

    return xpix_translated, ypix_translated

def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask

def rover_coords(thresholded_img):
    y0s, x0s = thresholded_img.nonzero()

    y_pixel = -(x0s - thresholded_img.shape[1]/2).astype('float')
    x_pixel = -(y0s - thresholded_img.shape[0]).astype('float')

    return x_pixel, y_pixel   

# Define a function to convert from cartesian to polar coordinates
def to_polar_coords(xpix, ypix):
    # Calculate distance to each pixel
    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    angles = np.arctan2(ypix, xpix)

    return dist, angles

def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Clip to world_size
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def find_rocks(img, levels=(110, 110, 50)):
    rockpix = (img[:, :, 0] > levels[0]) & (img[:, :, 1] > levels[1]) & (img[:, :, 2] < levels[2])
    rock = np.zeros_like(img[:, :, 0])
    rock[rockpix] = 1
    return rock

# Apply the above functions in succession and update Rover state accordingly
def perception_step(Rover):
    # 1. Define the source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                ])

    # 2. Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)

    # 3. Apply color threshold to identify navigable terrain/ obstacles/ rock samples
    colorsel = color_thresh(warped, lower_rgb_thresh=(160, 160, 160))
    obs_map = np.absolute(np.float32(colorsel) - 1) * mask

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    
    Rover.vision_image[:, :, 2] = colorsel * 255
    Rover.vision_image[:, :, 0] = obs_map * 255

    # 5. Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(colorsel)

    # 6. Convert rover-centric pixel values to world coords
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world = pix_to_world(obsxpix, obsypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size,scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[y_world, x_world, 2] += 10
    Rover.worldmap[obs_y_world, obs_x_world, 0] += 1

    # 8. Convert rover-centric pixel positions to polar coords
    dist, angles = to_polar_coords(xpix, ypix)

    # Update Rover pixel distances and angles
    Rover.nav_angles = angles

    # See if we can find some rocks
    rock_map = find_rocks(warped, levels=(110, 110, 50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)

        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx = rock_dist.argmin()
        rock_xcen, rock_ycen = rock_x_world[rock_idx], rock_y_world[rock_idx]

        Rover.worldmap[rock_ycen, rock_xcen] = 255
        Rover.vision_image[:, :, 1] = rock_map * 255
    else:
        Rover.vision_image[:, :, 1] = 0
    
    return Rover

def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

image = mpimg.imread('sample.jpg')
# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])