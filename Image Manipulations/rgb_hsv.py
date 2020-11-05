__author__ = 'Rajkumar Pillai'

import numpy as np
import cv2

"""
Description: This program is used to convert rgb to hsv and vice a versa and increase the saturation of the original image
"""

def clampIm(m,img):
    '''
    This function is used to create the image after claping at the maximum value m
    :param img:  The original image
    :param m:  The maximum value
    :return:
    '''

    m=m*255 ## Clamping value is percentage of 255
    rows, cols, channel = img.shape
    ClampedImage = np.zeros((rows, cols, 3), np.uint8)         #Creating a blank image to store the image after clamping

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

    return ClampedImage

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

    return shiftImage


def rgbhsv(img, k):
    '''
    This function genarates the new image which is increased saturation of original image
    :param img: The original image
    :param k: The constant value
    :return:
    '''

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    shiftImage = shiftIm(1, k, hsv_img)
    clampedImage = clampIm(255, shiftImage)

    hsv2rgb = cv2.cvtColor(clampedImage, cv2.COLOR_HSV2BGR)

    cv2.imshow('hsv-img', hsv2rgb)                           # Displaying the image
    cv2.waitKey(0)
    cv2.imwrite('Alice-rgb-hsv.png', hsv2rgb)                # Saving the image


def main():
    '''
    The main function
    :return:
    '''
    img = cv2.imread('Alice.jpg')                      # Reading the image
    rgbhsv(img,0.2)

if __name__ == '__main__':
    main()
