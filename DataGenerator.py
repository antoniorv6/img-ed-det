import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split 
from Logger import *
import random

CONST_TEST_FOLDER = "Test/Generator/"

bitwise_names = [   "FlipH", 
                    "FlipV", 
                    "Zoom", 
                    "Blur", 
                    "Inversion",
                    "Erosion",
                    "Dilation",
                    "Rot90",
                    "Rot180",
                    "None"]
class DataGen:
    def __init__(self):
        images = os.listdir("Data")

        self.images_list, self.validation_list = train_test_split(images, test_size=0.25, shuffle=True, random_state=1)

        self.images_index = 0
        self.val_images_index = 0
        self.bitwise_methods = [self.flip_horizontal, 
                                self.flip_vertical, 
                                self.zoom, 
                                self.blur, 
                                self.invert,
                                self.erode,
                                self.dilate,
                                self.rotate90,
                                self.rotate180,
                                self.none]
    
    def read_image(self, idxToRead, imagelist):
        return cv2.imread(f"Data/{imagelist[idxToRead]}")
    
    def flip_horizontal(self,image):
        return cv2.flip(image, 0)

    def flip_vertical(self,image):
        return cv2.flip(image, 1)
    
    def zoom(self,image):
        image_size = image.shape
        scaled_up_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        half_image = int(image_size[1]/2)
        three_quarter = int(image_size[1]/2) + image_size[1]
        return scaled_up_image[half_image:three_quarter, half_image:three_quarter]
    
    def blur(self, image):
        return cv2.GaussianBlur(image, (7,7), 0)
    
    def invert(self, image):
        return 255. - image

    def erode(self, image):
        kernel = np.ones((3,3), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def dilate(self, image):
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def rotate90(self, img):
        return cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    def rotate180(self, img):
        return cv2.rotate(img, cv2.cv2.ROTATE_180)
    
    def none(self, image):
        return image 

    def reset(self):
        self.images_index = 0
    
    def train_batch(self, BATCH_SIZE):
        X_source = []
        X_target = []
        Y = []
        image = None
        for idx_read in range(self.images_index, self.images_index+BATCH_SIZE):
            image = self.read_image(idx_read, self.images_list)
            edited_image = image
            edition_process = [np.random.binomial(1,0.5) for _ in range(10)]
            # First edition pass
            for idx in range(len(edition_process)):
                if edition_process[idx]:
                    edited_image = self.bitwise_methods[idx](edited_image)
            
            edition_to_detect = random.randint(0, len(self.bitwise_methods)-1)

            edit = self.bitwise_methods[edition_to_detect](edited_image)

            X_source.append(edited_image)
            X_target.append(edit)

            gt = np.zeros(len(self.bitwise_methods))
            gt[edition_to_detect] = 1.

            Y.append(gt)

        # Reset the index if we reach end of images list
        self.images_index += BATCH_SIZE
        if self.images_index + BATCH_SIZE > len(self.images_list):
            self.images_index = 0

        return np.array(X_source), np.array(X_target), np.array(Y)
    
    def val_batch(self, BATCH_SIZE):
        X_source = []
        X_target = []
        Y = []
        image = None
        ## READ BATCH SIZE IMAGES
        for idx_read in range(self.val_images_index, self.val_images_index+BATCH_SIZE):
            image = self.read_image(idx_read, self.validation_list)
            edited_image = image
            edition_process = [np.random.binomial(1,0.5) for _ in range(5)]
            for idx in range(len(edition_process)):
                if edition_process[idx]:
                    edited_image = self.bitwise_methods[idx](edited_image)
            
            edition_to_detect = random.randint(0, len(self.bitwise_methods)-1)

            edit = self.bitwise_methods[edition_to_detect](edited_image)

            X_source.append(edited_image)
            X_target.append(edit)

            gt = np.zeros(len(self.bitwise_methods))
            gt[edition_to_detect] = 1.

            Y.append(gt)

        # PERFORM RANDOMLY THE FOUR OPERATIONS, EDIT THE IMAGE AND SET BITWISE RESULT
        # ADD TO THE X AND Y ARRAYS

        # Reset the index if we reach end of images list
        self.val_images_index += BATCH_SIZE
        if self.val_images_index + BATCH_SIZE > len(self.validation_list):
            self.val_images_index = 0

        return np.array(X_source), np.array(X_target), np.array(Y)

    def get_val_size(self):
        return len(self.validation_list)

def main():
    dataGen = DataGen()
    BATCH_SIZE = 16
    dataGen.reset()
    X_source = [] 
    X_target = [] 
    Y = []

    DATA_GEN_LOG_INFO("CREATING TEST FOLDER")
    try:
        os.mkdir(CONST_TEST_FOLDER)
        DATA_GEN_LOG_INFO("TEST FOLDER CREATED")
    except OSError:
        DATA_GEN_LOG_WARNING("TEST FOLDER ALREADY IN DIRECTORY")

    DATA_GEN_LOG_INFO(f"GENERATING {1157} SAMPLES")

    for elx in range(1157):
        X_source, X_target, Y = dataGen.train_batch(BATCH_SIZE)
        for i,element in enumerate(Y):
            #imageName = ' '.join([str(elem) for elem in element]) + f"_{i}"
            for idxed, bit in enumerate(element):
                if bit == 1.0:
                    break
            
            cv2.imwrite(CONST_TEST_FOLDER + str(elx + i) + "_" + bitwise_names[idxed] +  "_src.jpg", X_source[i])
            cv2.imwrite(CONST_TEST_FOLDER + str(elx + i) + "_" + bitwise_names[idxed]  + "_tar.jpg", X_target[i])
    
    DATA_GEN_LOG_INFO(f"IMAGES GENERATED AND WRITTEN IN {CONST_TEST_FOLDER}")
    
    pass
    

if __name__ == "__main__":
    main()