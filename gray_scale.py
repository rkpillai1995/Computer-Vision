__author__ = 'Rajkumar Pillai'
import numpy as np
import cv2


"""
Description: This program converts original color swatch image to grayscale using weighted mean and adpative threshold technique
"""


def grayscale_color_swatch():
    '''
    This method is used to generate the grayscale image of original image
    :return:
    '''
    img = cv2.imread('color_swatch.jpg')               # Reading the image
    rows, cols, channel = img.shape


    blank_image_mean = np.zeros((rows, cols, 3), np.uint8)      # Creating a blank image to store the weighted mean grayscale version of orignal image
    blank_image_relative_luminance = np.zeros((rows, cols, 3), np.uint8) ## Creating a blank image to store the relative luminance grayscale version of orignal image

    ## Traversing through the pixels
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            b, g, r = k[0], k[1], k[2]
            gray_value_mean = int((0.33*r) + (0.33*g) + (0.33*b))      ## Weighted mean calculation

            gray_value_relative_luminance = int(0.299 * r + 0.587 * g + 0.114 * b)  ## Realtive mean calculation

            ##Storing the pixels for grayscale image
            blank_image_mean[i, j] = (gray_value_mean, gray_value_mean, gray_value_mean)
            blank_image_relative_luminance[i, j] = (gray_value_relative_luminance, gray_value_relative_luminance, gray_value_relative_luminance)

    cv2.imshow('weighted-mean', blank_image_mean)               # Displaying the image
    cv2.waitKey(0)
    cv2.imwrite('weighted-mean.png', blank_image_mean)           # saving the image


    cv2.imshow('realtive-luminance-mean', blank_image_relative_luminance)           # Displaying the image
    cv2.waitKey(0)
    cv2.imwrite('realtive-luminance-mean.png',blank_image_relative_luminance)       # saving the image





def main():

    '''
    The main function
    :return:
    '''
    grayscale_color_swatch()


if __name__ == '__main__':
    main()