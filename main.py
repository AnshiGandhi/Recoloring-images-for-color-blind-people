import tensorflow as tf
import cv2
import imutils
from imutils import paths
import numpy as np
import pandas as pd
import random
from ContrastBrightness import ContrastBrightness
from kmeans import Clusterer
import os
from skimage.morphology import skeletonize
from sklearn.metrics import classification_report
import streamlit as st
from PIL import Image
from pathlib import Path
import os, shutil

# Clearing the inputs dirsectory
folder = 'charts\input' 
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        st.write('Failed to delete %s. Reason: %s' % (file_path, e))

def load_image(image_file):
	img = Image.open(image_file)
	return img

# File Upload and save
upload_file = st.file_uploader("Choose a file",accept_multiple_files = True)
if upload_file is not None:
    for image_file in upload_file:
        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        # To View Uploaded Image
        st.image(load_image(image_file),width=250)
        # Save Image
        with open(os.path.join("charts\input",image_file.name),"wb") as f:
            f.write((image_file).getbuffer())
            st.success("File Saved")

CONTRASTER = ContrastBrightness()
CLUSTERER = Clusterer()

# used later as inputs to the neural network model
processed_images = []
image_labels = []
model = None

# prepares image paths and randomizes them
# image_paths = list(paths.list_images("charts/ordered"))
image_paths = list(paths.list_images("charts\input"))
# st.write(image_paths)
random.shuffle(image_paths)

# thresholding
def white_percent(img):
    w, h = img.shape
    total_pixels = w * h
    white_pixels=0
    for r in img:
        for c in r:
            if c == 255:
                white_pixels += 1
    return white_pixels/total_pixels

def fix_image(img):
    # inversion
    img = cv2.bitwise_not(img) 
    
    # thresholding
    image_bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]
        
    # making mask of a circle
    black = np.zeros((250,250)) # our images are 250x250 when we are transforming them
    # circle center is (125, 125), radius is 110, color is white
    # we divide by 255 to get image of 1s and 0s. Where a 0 is seen, our image will become black, which will
    # make our white outside black and keep the white digit as it is inside the circle
    circle_mask = cv2.circle(black, (125, 125), 110, (255, 255, 255), -1) / 255.0 
    
    # applying mask to make everything outside the circle black
    edited_image = image_bw * (circle_mask.astype(image_bw.dtype))
    return edited_image

for imagePath in image_paths:
    image = cv2.imread(imagePath)
    
    # resize
    image = imutils.resize(image, height=250)
       
    # contrast (no change in brightness)
    # Increase the contrast in order to make the color of the digit be more visible.
    image = CONTRASTER.apply(image, 0, 60)

    # show the image
    cv2.imshow("Gray", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # Apply median and Gaussian blurring to smooth out the small circles that we see in the original images.
    # blurring
    image = cv2.medianBlur(image,15)
    image = cv2.GaussianBlur(image,(3,3), cv2.BORDER_DEFAULT)

    # show the image
    cv2.imshow("Gray", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # color clustering
    image = CLUSTERER.apply(image, 5)

    # show the image
    cv2.imshow("Gray", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # (finally) grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # show the image
    cv2.imshow("Gray", gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # thresholding
    threshold = 0
    percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])
    while (not (percent_white > 0.10 and percent_white < 0.28)) and threshold <= 255:
        threshold += 10
        if threshold > 255:
            image_bw = fix_image(gray)
        else:
            image_bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # blurring to help remove noise
    image_bw = cv2.medianBlur(image_bw,7)
    image_bw = cv2.GaussianBlur(image_bw,(31,31),0)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # convert back to black and white after blurring
    image_bw = cv2.threshold(image_bw, 150, 255, cv2.THRESH_BINARY)[1]

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # apply morphology close
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # apply morphology open
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_OPEN, kernel)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # erosion (to make it thinner)
    kernel = np.ones((7,7), np.uint8)
    image_bw = cv2.erode(image_bw, kernel, iterations=1)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # skeletonizing
    image_bw = cv2.threshold(image_bw,0,1,cv2.THRESH_BINARY)[1]
    image_bw = (255*skeletonize(image_bw)).astype(np.uint8)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # dilating
    kernel = np.ones((21,21), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_DILATE, kernel)

    # show the image
    cv2.imshow("Gray", image_bw)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # append our finished image to the list (resize to 28x28 because our neural network needs those dimensions)
    processed_images.append(imutils.resize(image_bw, height=28))
    # st.write(type(processed_images))

    # # extract the correct label from the path of the file (because images are in folders by digit)
    # num = os.path.split(imagePath)[1][0:2]
    # image_labels.append(int(num))

    st.image(image_bw,width=250)

    # model loading
    model = tf.keras.models.load_model("mnist.h5")

    # reshaping our images to correct dimensions
    processed_images1 = np.array(processed_images)
    processed_images1 = processed_images1.reshape(processed_images1.shape[0], 28, 28, 1)
    processed_images1 =tf.cast(processed_images1, tf.float32)

    # image_labels = np.array(image_labels)

# making predictions and using np.argmax to convert the long vector output into a digit output
if(model):
    preds = np.argmax(model.predict(processed_images1), axis=1)
    st.write(preds)

# printing accuracy and other information
# st.write(classification_report(image_labels, preds))