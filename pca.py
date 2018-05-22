import numpy as np
import threading
import time
from collections import Counter
from numpy.linalg import linalg
import cv2


# Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a
# set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called
# principal components.
class PCA(object):
    def __init__(self, k_reduction, knn):
        self.labels = None
        self.U = None
        self.eigenFace = None
        self.k_reduction = k_reduction
        self.knn = knn

    # Given a set of images and labels, the training function evaluates the new reduced data-set and the matrix U
    def train(self, images, labels):
        self.labels = labels
        # self.num_labels = len(labels)
        # transform list of images to a matrix of witch every row=image. size of N observations each of M variables.
        # In our case M x 2500
        img_marix = np.asarray([image.flatten() for image in images], 'f')

        # Running the PCA algorithm, that reduces the dimensionality of the data-set.
        # Returned value is a matrix of k eigenvectors.
        self.U = pca_algorithm(self, img_marix)

        # saving a reduced size data-set, built from the original data-set and U.
        # eigenFace = #img_marix x U^t. of dimensions (N x K)
        self.eigenFace = np.matmul(img_marix, np.transpose(self.U))

    # Given an image, the function predict evaluates the label and the confidence of the image.
    # Using the knn algorithm that the given image is closest to
    def predict(self, image):
        # transform list of image to a matrix
        image = np.matmul(image.flatten(), np.transpose(self.U))

        # Running the KNN algorithm
        label, conf = knn(self, image)

        return label, conf


stop_thread = False


# def print_time():
#     global stop_thread
#     counter = 0
#     win_name = "counter"
#     cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#
#     while not stop_thread:
#         index = min(counter * 5, 100)
#         link = 'percentage/prcent{}.jpg'.format(index)
#         # link = 'maxr.jpg'
#         frame = cv2.imread(link, 1)
#         cv2.putText(frame, "The algorithm done {}% of it's work".format(counter * 5), (95, 85),
#                     cv2.FONT_HERSHEY_SIMPLEX, .7, (155, 55, 55), 2)
#
#         cv2.imshow(win_name, frame)
#         cv2.waitKey(1)
#         print("The algorithm done {}% of it's work".format(counter * 5))
#         time.sleep(3)
#         counter += 1
#     time.sleep(1)
#     cv2.destroyAllWindows()

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar
from PyQt5.QtCore import QBasicTimer
step_size = 1

def progress_bar():
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.progressBar = QProgressBar(self)
        self.progressBar.setGeometry(0, 0, 700, 200)
        self.setWindowTitle(" ")

        self.timer = QBasicTimer()
        self.step = 0
        self.timer.start(100, self)

    def timerEvent(self, event):
        if self.step >= 100:
            self.timer.stop()
            self.close()
            return
        self.step = self.step + step_size
        self.progressBar.setValue(self.step)


# Given a matrix of images,returns k highest eigenvectors.
#  k_reduction = the constant the is the num of eignvectors returned.
def pca_algorithm(self, original_data):
    global stop_thread
    data = np.matmul(original_data.transpose(), original_data)
    # w, v = [], []
    # pri_time = threading.Thread(name='print_time', target=print_time)
    # pri_time.start()
    # print("{}".format(original_data.shape[0] * original_data.shape[1]))
    # exit(0)
    data_size = original_data.shape[0] * original_data.shape[1]
    global step_size
    step_size = 100 / (1 + data_size / 490)
    progress_thread = threading.Thread(name='progress_bar', target=progress_bar)
    progress_thread.start()

    # Compute the eigenvalues \lambda_{i} and eigenvectors v_{i} of
    w, v = linalg.eig(data)
    stop_thread = True
    print("finish culc")
    # pri_time.join()
    progress_thread.join()

    # Sort the highest eigenvalues
    argwsort = np.argsort(w)[-self.k_reduction:][::-1]

    # From the highest eigenvalues chose k eigenvectors
    u = [list(value) for (i, value) in enumerate(v) if i in argwsort]

    return u


# K-nearest neighbors.
# In k-NN classification, the output is a class membership.
# An object is classified by a majority vote of its neighbors, with the object being assigned to the class most
# common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is
# simply assigned to the class of that single nearest neighbor.
def knn(self, original_img):
    distances_list = []
    for source_img in self.eigenFace:
        distance = np.sqrt(sum((source_img - original_img) ** 2))
        distances_list.append(distance)

    kNeighboursIndex = np.argsort(distances_list)[:self.knn]

    kNeighboursLabel = [self.labels[index] for index in kNeighboursIndex]

    label, count = Counter(kNeighboursLabel).most_common()[0]

    return label, (count / self.knn)
