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
    processed_image()

# Processing of images takes place here.
def processed_image():
    for index, img in enumerate(images):
        '''
        Converting RGB to LAB to process luminosity 
        without effecting color channel i.e A and B 
        '''
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        '''
        L - light axis which controls the luminance
        a axis - color channel (green to red)
        b axis - color channel (blue to yellow)
        '''
        l, a, b = cv2.split(lab)
        '''
        Applying CLAHE to L channel to equalize images 
        in order to improve contrast
        '''
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        '''
        Merging channels - L, A, B
        '''
        limg = cv2.merge((cl,a,b))
        '''
        Converting LAB to RGB
        '''
        processed_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        ''' 
        Saving output images on folder 
        '''
        if not os.path.exists("Processed_Images"):
            os.mkdir("Processed_Images")
            cv2.imwrite("./Processed_Images/output_"+str(index)+".jpg", processed_img)
        else:
            cv2.imwrite("./Processed_Images/output_"+str(index)+".jpg", processed_img)

if __name__ == "__main__":
    main()