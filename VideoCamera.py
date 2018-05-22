import cv2


class VideoCamera(object):
    def __init__(self, url=0):
        self.video = cv2.VideoCapture(url)
        self.url = url

    def __del__(self):
        self.video.release()

    def get_frame(self, in_gray_scale=False):
        _, frame = self.video.read()
        if in_gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
