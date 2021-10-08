# Importing Libraries
import cv2
import os

# Empty list for reading and appending images.
images = []

# Reading images.
def main():
    for i in os.listdir('./Fog'):
        i = "./fog/"+i
        if ".jpg" in i or ".png" in i or ".jpeg" in i :
            img = cv2.imread(i,1)
            if (img is None):
                print("Image not read properly",i)
            images.append(cv2.imread(i, 1))