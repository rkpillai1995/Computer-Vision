__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import cv2


"""
Description: This program generates Histogram of sonnet.png and modified image using global threshold and adaptive thresholding.
"""

def main():
    '''
    The main function
    :return:
    '''

    ##Plotting the histogram of image
    img = cv2.imread('sonnet.png', 0)
    hist=plt.hist(img.ravel(), 256, [0, 256]);
    plt.title('Histogram of original image')
    plt.savefig('Histogram_sonnet.png')
    plt.show()
    print(hist)

    global_threshold_value=130  ## A global threshold value which was manually determined from histogram


    ##These below code can be uncommented to compute the global threshold value using Otsu's method
    '''
    global_threshold_value,thresholded_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print('Global Threshold Value',global_threshold_value)
    plt.imshow(thresholded_image, 'gray')
    plt.show()
    '''

    ## To modify the image based on global threshold value
    global_threshold_value,thresholded_image = cv2.threshold(img, global_threshold_value, 255, cv2.THRESH_BINARY)
    print('Global Threshold Value',global_threshold_value)
    plt.title('Results at a global threshold value')
    plt.imshow(thresholded_image, 'gray')
    plt.savefig('Modified_Image_Global_Threshold_value.png')
    plt.show()

    ## To modify the image using adaptive thresholding using mean of neighborhood pixel
    adaptive_threshold_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11,7 )
    plt.title('Results using adaptive thresholding')
    plt.imshow(adaptive_threshold_image, 'gray')
    plt.savefig('Modified_Image_Adaptive_Threshold_value.png')
    plt.show()


if __name__ == '__main__':
    main()