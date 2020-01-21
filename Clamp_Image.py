__author__ = 'Rajkumar Pillai'

import numpy as np
import cv2


"""
Description: This program clamps the input image at a value m if the channel exceeds else clamps at zero
"""

def clampIm(img,m):
    '''
    This function is used to create the image after claping at the maximum value m
    :param img:  The original image
    :param m:  The maximum value
    :return:
    '''

    m=m*255 ## Clamping value is percentage of 255
    rows, cols, channel = img.shape
    ClampedImage = np.zeros((rows, cols, 3), np.uint8)         # Creating a blank image to store the image after clamping

    ## Traversing through the pixels
    for i in range(rows):
        for j in range(cols):
            Pixel_value = img[i, j]
            b, g, r = Pixel_value[0], Pixel_value[1], Pixel_value[2]

        ## Cheking if the value in each channel exceeds m or is it less than zero
            if b > m:
                clamp_value_b = m
            elif b<0:
                clamp_value_b =0
            else:
                clamp_value_b = b
            if g > m:
                clamp_value_g = m
            elif g<0:
                clamp_value_g = 0
            else:
                clamp_value_g = g
            if r > m:
                clamp_value_r = m
            elif r<0:
                clamp_value_r = 0
            else:
                clamp_value_r = r
            ClampedImage[i, j] = (clamp_value_b, clamp_value_g, clamp_value_r)   ## Storing the pixels for modified image

    cv2.imshow('clamped-image', ClampedImage)                       # Displaying the image
    cv2.waitKey(0)
    cv2.imwrite('Alice-clamp.png', ClampedImage)                    # Saving the image


def main():
    '''
    The main function
    :return:
    '''
    img = cv2.imread('Alice.jpg')           # Reading the image
    clampIm(img,0.45)


if __name__ == '__main__':
    main()