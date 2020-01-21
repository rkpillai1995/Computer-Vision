
import numpy as np
import cv2



def BRGB(img):

    b = img.copy()
    b[:, :, 0] = 0
    cv2.imshow('B-RGB', b)
    cv2.waitKey(0)



def grayscale_color_swatch():
    img = cv2.imread('color_swatch.jpg')
    '''
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    grayscale_weighted_mean = ((r+g+b)/3)
    grayscale_relative_luminance = 0.299 * r + 0.587 * g + 0.114* b

    cv2.imshow('weighted-mean', grayscale_weighted_mean)
    cv2.waitKey(0)
    cv2.imshow('realtive-luminance-mean', grayscale_relative_luminance)
    cv2.waitKey(0)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('realtive-luminance-mean', gray_image)
    cv2.waitKey(0)
    '''
    rows,cols,channel=img.shape
    print(rows,cols)
    blank_image_mean = np.zeros((rows, cols, 3), np.uint8)
    blank_image_relative_luminance = np.zeros((rows, cols, 3), np.uint8)

    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            b, g, r = k[0], k[1], k[2]
            gray_value_mean=int((r+g+b)/3)
            #print(gray_value_mean)
            gray_value_relative_luminance=int(0.299 * r + 0.587 * g + 0.114* b)
            blank_image_mean[i,j]=(gray_value_mean,gray_value_mean,gray_value_mean)
            blank_image_relative_luminance[i,j]=(gray_value_relative_luminance,gray_value_relative_luminance,gray_value_relative_luminance)
    cv2.imshow('weighted-mean', blank_image_mean)
    cv2.waitKey(0)
    cv2.imshow('realtive-luminance-means', blank_image_relative_luminance)
    cv2.waitKey(0)

def shiftIm(img_Channel,constant,img):
    rows, cols, channel = img.shape
    shiftImage = np.zeros((rows, cols, 3), np.uint8)


    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            if img_Channel == 0:

               b, g, r = k[0], k[1], k[2]
               #print(b)
               shift_value=int(b+constant)
               shiftImage[i, j]=(shift_value,g,r)
            if img_Channel == 1:

               b, g, r = k[0], k[1], k[2]
               #print(b)
               shift_value=int(g+constant)
               shiftImage[i, j]=(b,shift_value,r)
            if img_Channel == 2:

               b, g, r = k[0], k[1], k[2]
               #print(b)
               shift_value=int(r+constant)
               shiftImage[i, j]=(b,g,shift_value)

    cv2.imshow('shifted-image', shiftImage)
    cv2.waitKey(0)
    return shiftImage
def clampIm(constant,img):
    rows, cols, channel = img.shape
    ClampedImage = np.zeros((rows, cols, 3), np.uint8)
    #b = img.copy()
    #print(b[:, :, ])

    for i in range(rows):
        for j in range(cols):
               k = img[i, j]
               b, g, r = k[0], k[1], k[2]
            
               if b > constant or g >constant or r > constant:
                    clamp_value_b=b+constant
                    clamp_value_g =g+constant
                    clamp_value_r =r+  constant
                    ClampedImage[i, j]=(clamp_value_b,clamp_value_g,clamp_value_r)
               if b <0 or g<0 or r<0:
                   print('reach')
                   clamp_value =0
                   ClampedImage[i, j] = (clamp_value, clamp_value, clamp_value)
    cv2.imshow('clamped-image', ClampedImage)
    cv2.waitKey(0)

def clampImnew(constant):
    img = cv2.imread('Alice.jpg')
    rows, cols, channel = img.shape
    ClampedImage = np.zeros((rows, cols, 3), np.uint8)
    image=img.copy()
    print("clampim")
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            b, g, r = k[0], k[1], k[2]

            if b > constant  :
                clamp_value_b =  constant
            else:
                clamp_value_b=b
            if  g > constant:
                clamp_value_g = constant
            else:
                clamp_value_g=g
            if r>constant:
                clamp_value_r =  constant
            else:
                clamp_value_r = r
            ClampedImage[i, j] = (clamp_value_b, clamp_value_g, clamp_value_r)
            if b < 0 or g < 0 or r < 0:
                print('reach')
                clamp_value = 0
                ClampedImage[i, j] = (clamp_value, clamp_value, clamp_value)
    cv2.imshow('clamped-image', ClampedImage)
    cv2.waitKey(0)
    '''
    ClampedImagenew = np.zeros((rows, cols, 3), np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            k = img[i, j]
            b, g, r = k[0], k[1], k[2]
            maxvalue=max(b,max(g,r))
            if b > maxvalue:
                clamp_value_b = maxvalue
            else:
                clamp_value_b=b
            if g > maxvalue:
                clamp_value_g = maxvalue
            else:
                clamp_value_g=g

            if r > maxvalue:
                clamp_value_r = maxvalue
            else:
                clamp_value_r = r

            ClampedImagenew[i, j] = (clamp_value_b, clamp_value_g, clamp_value_r)
    cv2.imshow('clamped-image', ClampedImagenew)
    cv2.waitKey(0)

    cv2.imshow('clamped-image', img)'''

    #cv2.waitKey(0)

def rgbhsv(img,value):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv_img', hsv_img)

    #cv2.waitKey(0)
    shiftImage=shiftIm(1,value,hsv_img)
    hsv2rgb=cv2.cvtColor(shiftImage, cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv_img', hsv2rgb)

    cv2.waitKey(0)

def main():
    img = cv2.imread('Alice.jpg')

    BRGB(img)
    grayscale_color_swatch()
    shiftIm(2,85,img)
    ###clampIm(120,img)
    clampImnew(120)
    rgbhsv(img,80)
if __name__ == '__main__':
    main()