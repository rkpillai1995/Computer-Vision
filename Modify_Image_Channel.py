__author__ = 'Rajkumar Pillai'

import numpy as np
import cv2


"""

Author: Rajkumar Lenin Pillai

Description: This program is used to add a constant value to an image channel
"""

def shiftIm(img_Channel, k, img):
    '''
    This function is used to add constant value to an image channel and generate the resulting image
    :param img_Channel: The integer denoting the image channel
    :param k:  The constant value
    :param img: The original image
    :return:
    '''


    k=k*255  ## Shifting value is percentage of 255
    rows, cols, channel = img.shape
    shiftImage = np.zeros((rows, cols, 3), np.uint8)

    ## Traversing through the pixels
    for i in range(rows):
        for j in range(cols):
            Pixel_vlaue = img[i, j]
            if img_Channel == 0:
               b, g, r = Pixel_vlaue[0], Pixel_vlaue[1], Pixel_vlaue[2]
               shifted_value=int(b + k)                                ## Adding constant value to the image channel
               shiftImage[i, j]=(shifted_value,g,r)                    ## Storing the pixels for modified image

            if img_Channel == 1:
               b, g, r = Pixel_vlaue[0], Pixel_vlaue[1], Pixel_vlaue[2]
               shifted_value=int(g + k)                               ## Adding constant value to the image channel
               shiftImage[i, j]=(b,shifted_value,r)                   ## Storing the pixels for modified image

            if img_Channel == 2:
               b, g, r = Pixel_vlaue[0], Pixel_vlaue[1], Pixel_vlaue[2]
               shifted_value=int(r + k)                              ## Adding constant value to the image channel
               shiftImage[i, j]=(b,g,shifted_value)                  ## Storing the pixels for modified image

    cv2.imshow('shifted-image', shiftImage)                       # Displaying the image
    cv2.waitKey(0)
    cv2.imwrite('Alice-Shift-Image.png', shiftImage)              # saving the image

def main():
    '''
    The main function
    :return:
    '''
    img = cv2.imread('Alice.jpg')       # reading the image

    shiftIm(1,0.45,img)

if __name__ == '__main__':
    main()