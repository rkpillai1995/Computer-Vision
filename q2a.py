__author__ = 'Rajkumar Pillai'

import cv2


"""
file: q2a.py
CSCI-631:  COMPUTER VISION
Author: Rajkumar Lenin Pillai

Description: This program generates Alice image with blue channels of original image set to zero
"""


def BRGB(img):
    '''
    Use to set the blue channels of original image to zero
    :param img: The original image
    :return:
    '''

    b = img.copy()               # Creating a copy of original  image
    b[:, :, 0] = 0               # Setting the blue channels to zero in the copy of original image
    cv2.imshow('B-RGB', b)       # Displaying the image
    cv2.waitKey(0)

    cv2.imwrite('Alice-no-blue.png', b)      # saving the image


def main():
    '''
    The main function
    :return:
    '''

    img = cv2.imread('Alice.jpg') # reading the image
    BRGB(img)


if __name__ == '__main__':
    main()