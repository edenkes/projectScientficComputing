import cv2


class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.3
        min_neighbor = 5
        min_size = (130, 130)
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else cv2.CASCADE_SCALE_IMAGE

        faces_coordinates = self.classifier.detectMultiScale(image, scaleFactor=scale_factor, minNeighbors=min_neighbor,
                                                          minSize=min_size, flags=flags)

        return faces_coordinates
