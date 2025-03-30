from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
from sklearn import svm
from sklearn import metrics

class SVM:
    namelabels = []
    # function to convert image to features
    def getFeatures(self,image):
        image = cv2.resize(image, (500, 500))
        # convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # compute the color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # normalize the histogram
        cv2.normalize(hist, hist)
        return hist.flatten()

    # Step1:  prepare training data
    def prepareTrainingData(self,folderPath):
        images = []
        labels = []
       
        #Read each image
        for directory in os.listdir(folderPath):
            if directory.startswith("."):
                continue

            # read each image
            for imageName in os.listdir(folderPath+"/"+directory):

                if imageName.startswith("."):
                    continue
            
                
                imagePath = folderPath +"/"+directory + "/" + imageName
                image = cv2.imread(imagePath)
                print(imagePath)
                images.append(self.getFeatures(image))
                labels.append(SVM.namelabels.index(directory))

        return images, labels
