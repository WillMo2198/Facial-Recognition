from PIL.ImageEnhance import Contrast
from PIL import Image
import numpy as np
import cv2
import os

from cv2 import waitKey

import ai123
from time import sleep

np.set_printoptions(suppress=False)


class Recognizer:
    def __init__(self, name=None):
        self.cap = cv2.VideoCapture(0)
        _, self.screen = self.cap.read()
        self.haar_cascade = ai123.Datasets().haar_cascade
        self.name = name

    def get_roi(self, faces): # Select reigon of intrest
        self.screen = np.asarray(self.screen)
        for (x, y, w, h) in faces:
            roi = self.screen[y:y + h, x:x + w]
            roi = Image.fromarray(roi).resize((75, 75))
            roi = Contrast(roi).enhance(2)  # Increase contrast for better color mapping
            roi = cv2.cvtColor(np.asarray(roi), cv2.COLOR_BGR2GRAY)  # Convert screen to black and white
            screen = cv2.rectangle(np.asarray(self.screen), (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rect around ROI
            return screen, roi

    def detect(self):
        faces = self.haar_cascade.detectMultiScale(self.screen)  # Use haar cascade to detect any faces present
        return faces

    def record(self):
        _, self.screen = self.cap.read()
        faces = self.detect()
        self.screen, roi = self.get_roi(faces)
        return self.screen, roi


class User:
    def __init__(self, name):
        self.name = name
        self.recognizer = Recognizer(name)

    def new_user(self):
        if not os.path.exists(f'Users/{self.name}'):    # Make directory for new user training data
            os.makedirs(f'Users/{self.name}')
            sleep(1)
        print('Press "space" to quit.')
        while len(os.listdir(f'Users/{self.name}')) < 100:
            try:
                screen, roi = self.recognizer.record()
            except TypeError:   # If no faces found, keep loop going
                continue
            cv2.imshow('Scan', np.asarray(screen))
            cv2.imshow('ROI', np.asarray(roi))
            """Image.fromarray(np.array(roi)).save('Users/{0}/{1}.jpg'.format(self.name, photo_i))"""
            cv2.waitKey(1)


print('Initalizing...')
User('name').new_user()
