__author__ = 'Rajkumar Pillai'

import  matplotlib.pyplot as plt
import cv2
import pathlib
import numpy as np
from sklearn.cluster import KMeans

"""
file: q1.py
CSCI-631:  COMPUTER VISION
Author: Rajkumar Lenin Pillai

Description: This program generates SIFT features of train images and Histogram of one image of category using bag of words model 
"""

def file_reader(path):
    '''
    This function is used to read the  training images in the given directory
    :param path: The path of folder which contains the train  images folder
    :return:list_Of_image_files_and_word    ## Dictionary used to store images of that corresponding category
           :No_of_image_files               ## Initializing no of image files
    '''

    # define the path
    currentDirectory = pathlib.Path(path)

    No_of_image_files=0             ## Initializing no of image files
    list_Of_image_files_and_word={} ## Dictionary used to store images of that corresponding category

    ## To iterate inside the train  folder
    for currentFile in currentDirectory.iterdir():
        path_name=str(currentFile)  ## To store the name of a category
        category=path_name.split('\\')[-1]

        currentdir=pathlib.Path(currentFile)

        # define the pattern
        currentPattern = "*.jpg"

        list_Of_image_files_and_word[category]=[]

        ## To iterate inside a particular category of image
        for files in currentdir.glob(currentPattern):
            image=cv2.imread(str(files),0)       ## Reading the image files that belong to a category
            No_of_image_files=No_of_image_files+1
            list_Of_image_files_and_word[category].append(image) ## Storing the image file corresponding to the category

    return list_Of_image_files_and_word,No_of_image_files

def SIFT_features_calculation(image):
    '''
    This function computes the SIFT features of the image provided
    :param image: The input image of which SIFT feature is to be computed
    :return: kp: keypoints
             des: Descriptors
    '''


    sift = cv2.xfeatures2d.SIFT_create()           ## Creating the SIFT object in cv2
    kp, des = sift.detectAndCompute(image, None)   ## Calling the SIFT feature detection function
    return kp, des


def train_model(training_path,cluster_size):

    '''
    This function is used to train the model using K-means
    :param training_path: The paht whcih contains the images for training
    :param cluster_size:  The no of clusters to be used
    :return: kmeans_model: The model which is trained on training data images
             cluster_size:
    '''

    print("Computing SIFT features of Images")
    ## Getting the dictionary of image for each category and total no of files
    list_Of_image_files_and_category, No_of_image_files=file_reader(training_path)

    descriptor_list=[]  ## To store descriptors for each image


    ## Iterating through every image of every category
    for category, images in list_Of_image_files_and_category.items():

        Image_belonging_to_category_visited=False        ## A flag used to show SIFT features of image in each category

        for image in images:
            kp, des = SIFT_features_calculation(image)   ## Calling SIFT function to get the keypoints and descriptors of image
            img = cv2.drawKeypoints(image, kp,None)      ## Drwaing the keypoints on image
            if Image_belonging_to_category_visited is False:  ## If none of image in this category are shown

                ## To show SIFT features of one image of each category
                Title="SIFT features of " + category + " category"
                plt.title(Title)
                plt.imshow(img)
                #plt.savefig(Title+".jpg")
                plt.show()
            Image_belonging_to_category_visited=True

            descriptor_list.append(des)              ## Storing the descriptors in a list

    ## To make the descriptor stack for all training images
    descriptor_stack=np.array(descriptor_list[0])
    secondlist=descriptor_list[1:]
    for items in secondlist:
        descriptor_stack=np.concatenate((descriptor_stack,items))

    ## Buidling the model using Kmeans
    print("Training using K-means")
    kmeans_obj= KMeans(n_clusters=cluster_size)
    kmeans_model = kmeans_obj.fit(descriptor_stack)

    return kmeans_model,cluster_size



def model_evaluation(path,kmeans_model,cluster_size):
    '''
    This function evalutes the model on the  images
    :param path: Th folder where  images are present
    :param kmeans_model: The trained k-means model
    :param cluster_size:  The size of the cluster
    :return:
    '''

    print("Plotting Histograms")
    ## Getting the dictionary of image for each category and total no of files
    list_Of_image_files_and_category, No_of_image_files = file_reader(path)

    ## Iterating through every image of every category
    for category, images in list_Of_image_files_and_category.items():

        Image_belonging_to_category_visited = False                 ## A flag used to show histogram of image in each category
        descriptor_list = []                           ## To store descriptors for each image

        for image in images:
            if Image_belonging_to_category_visited is False:
                kp, des = SIFT_features_calculation(image) ## Calling SIFT function to get the keypoints and descriptors of image
                descriptor_list.append(des)                ## Storing the descriptors in a list
                plot_histogram_of_images(descriptor_list,kmeans_model,category,cluster_size)  ##Calling function to plot histogram
                Image_belonging_to_category_visited = True


def plot_histogram_of_images(descriptor_list,kmeans_model,category,cluster_size):
    '''
    This function uses the model to predict  and plots the histogram of images
    :param descriptor_list:  To store descriptors for each image
    :param kmeans_model: The trained k-means model
    :param category: The name of the category
    :param cluster_size: The size of cluster
    :return:
    '''

    ## To make the descriptor stack for all testing images
    descriptor_stack = np.array(descriptor_list[0])
    secondlist = descriptor_list[1:]
    for items in secondlist:
        descriptor_stack = np.concatenate((descriptor_stack, items))

    ## Prediction using trained model
    hist = kmeans_model.predict(descriptor_stack)

    ## Plotting the histogram
    Title="Histogram of "+category+" category"
    plt.title(Title)
    plt.xticks(range(cluster_size))
    plt.ylabel("Frequency")
    plt.hist(hist)
    #plt.savefig(Title+".jpg")
    plt.show()


def main():
    '''
    The main function
    :return:
    '''

    training_path="images/train/" ## Path where training images are stored
    #testing_path="images/test/"   ## Path where testing images are stored
    cluster_size=100              ## Declaring cluster size

    kmeans_value,cluster_size=train_model(training_path,cluster_size) ## Calling the train function
    model_evaluation(training_path,kmeans_value,cluster_size)          ## Calling evaluation function


if __name__ == '__main__':
    main()