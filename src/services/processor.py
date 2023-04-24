import cv2
import os
import shutil
import glob
import tensorflow as tf
import numpy as np
import keras.utils as image
from flask import Flask, send_from_directory

from src.utils.flip import flipImages
from src.utils.outDoor_PoseEstimation import outDoor_PoseEstimation_F
from src.utils.inDoor_PoseEstimation import inDoor_PoseEstimation_F

app = Flask(__name__)

models_dir = os.path.abspath(os.path.join(app.root_path, os.pardir, 'models'))
assets_dir = os.path.abspath(os.path.join(app.root_path, os.pardir, 'assets'))

video_path = os.path.join(assets_dir, 'videoinput', 'video.MOV')

################################################################

ImageSkeletonStance_model = tf.keras.models.load_model(os.path.join(models_dir, 'ImageSkeleton_Stance.h5'))
ImageSkeletonLeg_model = tf.keras.models.load_model(os.path.join(models_dir, 'ImageSkeleton_Leg.h5'))
ImageSkeletonShot_model = tf.keras.models.load_model(os.path.join(models_dir, 'ImageSkeleton_Shot.h5'))

SkeletonOnlyStance_model = tf.keras.models.load_model(os.path.join(models_dir, 'skeletonOnly_Stance.h5'))
SkeletonOnlyLeg_model = tf.keras.models.load_model(os.path.join(models_dir, 'skeletonOnly_Leg.h5'))
SkeletonOnlyShot_model = tf.keras.models.load_model(os.path.join(models_dir, 'skeletonOnly_Shot.h5'))


################################################################

def ImageStance():
    class_names = loadClassNames()
    image_path = os.path.join(assets_dir, 'output', 'stance.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = ImageSkeletonStance_model.predict(test_image)

    stanceDetection1 = class_names[int(np.argmax(result, axis=1))]
    return stanceDetection1


def ImageLeg():
    class_names = loadClassNames()
    image_path = os.path.join(assets_dir, 'output', 'legMovement.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = ImageSkeletonLeg_model.predict(test_image)

    legDetection3 = class_names[int(np.argmax(result, axis=1))]
    return legDetection3


def ImageShot():
    class_names = loadClassNames()

    image_path = os.path.join(assets_dir, 'output', 'shotExecution.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = ImageSkeletonShot_model.predict(test_image)

    shotDetection5 = class_names[int(np.argmax(result, axis=1))]
    return shotDetection5


def SkeletonStance():
    class_names = loadClassNames()
    image_path = os.path.join(assets_dir, 'output', 'stance.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = SkeletonOnlyStance_model.predict(test_image)

    stanceDetection2 = class_names[int(np.argmax(result, axis=1))]
    return stanceDetection2


def SkeletonLeg():
    class_names = loadClassNames()
    image_path = os.path.join(assets_dir, 'output', 'legMovement.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = SkeletonOnlyLeg_model.predict(test_image)

    legDetection4 = class_names[int(np.argmax(result, axis=1))]
    return legDetection4


def SkeletonShot():
    class_names = loadClassNames()
    image_path = os.path.join(assets_dir, 'output', 'shotExecution.jpg')

    test_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
    test_image = image.img_to_array(test_img)
    test_image = np.expand_dims(test_image, axis=0)

    result = SkeletonOnlyShot_model.predict(test_image)

    shotDetection6 = class_names[int(np.argmax(result, axis=1))]
    return shotDetection6


def loadClassNames():
    class_names = ['correct', 'incorrect']
    return class_names


def processMovieAndCreateRelatedImages():
    folder_path1 = os.path.join(assets_dir, 'videooutput')
    files = glob.glob(os.path.join(folder_path1, '*'))
    for f in files:
        os.remove(f)

    folder_path2 = os.path.join(assets_dir, 'flipped')
    files = glob.glob(os.path.join(folder_path2, '*'))
    for f in files:
        os.remove(f)

    folder_path3 = os.path.join(assets_dir, 'output')
    files = glob.glob(os.path.join(folder_path3, '*'))
    for f in files:
        os.remove(f)


def generateResults(flipOrNot, indoorOrOutdoor):
    # Define the video file path and the output directory
    output_dir = os.path.join(assets_dir, 'videooutput')
    flipped_Input = os.path.join(assets_dir, 'flipped')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Get the video frame rate and resolution
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set the desired time points to extract frames from and their corresponding names
    time_points = {'stance': 0.4, 'legMovement': 1.9, 'shotExecution': 3.4}

    # Loop through the frames in the video and extract the desired frames as images
    for name, time_point in time_points.items():
        # Set the video frame position to the desired time point
        cap.set(cv2.CAP_PROP_POS_MSEC, time_point * 1000)
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to 3840x2160 resolution
        resized_frame = cv2.resize(frame, (3840, 2160))
        # Set the output path for the image
        output_path = os.path.join(output_dir, '{}.jpg'.format(name))
        # Save the image as a JPEG file with maximum quality
        cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    # Release the video file
    cap.release()

    stance = ""
    leg = ""
    shot = ""

    if flipOrNot == "Left":

        flipImages(output_dir, flipped_Input)  # callig flipping method

        # clear videooutput file to restore flliped images
        files = glob.glob(os.path.join(output_dir, '*'))
        for f in files:
            os.remove(f)

        # Get a list of all files in the source folder and copy them
        files = os.listdir(flipped_Input)
        for file in files:
            src_file = os.path.join(flipped_Input, file)
            dst_file = os.path.join(output_dir, file)
            shutil.copy(src_file, dst_file)

        if indoorOrOutdoor == "Out":
            outDoor_PoseEstimation_F(assets_dir)
            stance = ImageStance()
            leg = ImageLeg()
            shot = ImageShot()

        elif indoorOrOutdoor == "In":
            inDoor_PoseEstimation_F(assets_dir)
            stance = SkeletonStance()
            leg = SkeletonLeg()
            shot = SkeletonShot()


    elif flipOrNot == "Right":

        if indoorOrOutdoor == "Out":
            outDoor_PoseEstimation_F(assets_dir)
            stance = ImageStance()
            leg = ImageLeg()
            shot = ImageShot()

        elif indoorOrOutdoor == "In":
            inDoor_PoseEstimation_F(assets_dir)
            stance = SkeletonStance()
            leg = SkeletonLeg()
            shot = SkeletonShot()

    stanceProper = stance
    legProper = leg
    shotProper = shot

    return [stanceProper, legProper, shotProper]


def processVideo(hand, area):
    processMovieAndCreateRelatedImages()
    return generateResults(hand, area)


def saveFileLocally(file):
    # Save the file to the server
    save_path = os.path.join(assets_dir, 'videoinput', 'video.MOV')
    file.save(save_path)


def loadImage(filename):
    path = os.path.join(assets_dir, 'output')
    return send_from_directory(path, filename)
