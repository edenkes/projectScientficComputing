import os

import numpy as np

import cv2

from FaceDetector import FaceDetector
from VideoCamera import VideoCamera
from pca import PCA


# cut the images to include only the face
def cut_faces(image, faces_coordinates):
    faces = []

    for (top, right, bottom, left) in faces_coordinates:
        bottom_rm = int(0.2 * bottom / 2)
        faces.append(image[right:right + left, top + bottom_rm: top + bottom - bottom_rm])
    return faces


# normalize the rgb images to grey scale
def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Histogram - It is a graphical representation of the intensity distribution of an image.
        # Histogram Equalization - It is a method that improves the contrast in an image, in order to stretch out
        #                           the intensity range.
        images_norm.append(cv2.equalizeHist(image))     #
    return images_norm


# resize the image to 50 x 50
def resize(images, size=(50, 50)):
    images_norm = []
    for image in images:
        # Comparison of OpenCV Interpolation Algorithms:
        # INTER_NEAREST - a nearest-neighbor interpolation
        # INTER_LINEAR - a bilinear interpolation (used by default)
        # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation,
        #              as it gives moire’-free results. But when the image is zoomed, it is similar to the
        #              INTER_NEAREST method.
        # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
        # Bicubic interpolation - is an extension of cubic interpolation for interpolating data points on a
        # two-dimensional regular grid.
        # Cubic interpolition - given argemnts values x_1, ... , x_4 to obtain a smooth continuous function.
        # this is similar to Bizer curves but passes through all 4 points.
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) # piece wise cubic interpolation and bicubic
        images_norm.append(image_norm)
    return images_norm


# set the image to standard settings
def normalize_faces(image, faces_coordinates):
    faces = cut_faces(image, faces_coordinates)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

# the algorithm to adding a face image to the data-set
def build_data_set(url=0, win_name="live!"):
    # Face Detection using Haar Cascades:
    # Goal: We will see the basics of face detection using Haar Feature-based Cascade Classifiers
    # Basics: Each feature is a single value obtained by subtracting sum of pixels under the white rectangle from sum
    #         of pixels under the black rectangle.
    # OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc.
    # Those XML files are stored in the opencv/data/haarcascades/ folder.
    # It is a machine learning based approach where a cascade function is trained from a lot of positive and
    # negative images. It is then used to detect objects in other images.

    video = VideoCamera(url)
    detector = FaceDetector('haarcascade_frontalface_default.xml')
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    name = ""
    number_of_pic = 10

    print("Enter Your name")
    while True:
        frame = video.get_frame()
        height, width, _ = frame.shape

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(40) & 0xFF
        if key not in [8, 13, 27, 255]:
            name += chr(key)
            print(name)
        elif key == 8:
            name = name[:-1]
        elif key == 27:
            cv2.destroyAllWindows()
            return
        elif key == 13:
            folder = "people/" + name.lower()  # input name
            break

    if not os.path.exists(folder):
        os.mkdir(folder)
    init_pic = len(os.listdir(folder))
    counter = init_pic
    timer = 0
    while counter < number_of_pic + init_pic:  # take 10 photo
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        if len(faces_coordinates) and timer % 700 == 50:
            faces = normalize_faces(frame, faces_coordinates)
            cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0])
            counter += 1
        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        timer += 50
    cv2.destroyAllWindows()


# takes all images of every file in the folder people.
# Creates for every sub-file an array of the images and labels and a mapping of them.
def collect_dat_set():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0))
            labels.append(i)
    return images, np.array(labels), labels_dic


# Trains the set of images
def train_models():
    images, labels, labels_dic = collect_dat_set()

    rec_eig = PCA(500, 5)
    if images:
        rec_eig.train(images, labels)

    return rec_eig, labels_dic


# given an image returns the name of the face it resembles most.
def face_recognition(url=0, win_name="live!"):
    rec_eig, labels_dic = train_models()
    print("Finished training")

    video = VideoCamera(url)
    frame = video.get_frame()

    detector = FaceDetector('haarcascade_frontalface_default.xml')
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    threshold = 0.5

    while True:
        frame = video.get_frame()
        faces_coordinates = detector.detect(frame)
        if len(faces_coordinates):
            faces = normalize_faces(frame, faces_coordinates)
            for i, face in enumerate(faces):
                pred, conf = rec_eig.predict(faces[i])
                if conf > threshold:
                    print("Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))
                    cv2.putText(frame, labels_dic[pred].capitalize(),
                                (100, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                                (76, 63, 243), 2)
                else:
                    print("Low Prediction: {}, Confidence: {}.".format(labels_dic[pred].capitalize(), conf))
        cv2.imshow(win_name, frame)

        if cv2.waitKey(1) & 0xff == 27:
            break

    video.__del__()
    cv2.destroyAllWindows()