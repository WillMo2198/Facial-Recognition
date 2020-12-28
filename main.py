import numpy as np
from PIL import Image, ImageEnhance
from os import listdir, mkdir
from random import randint
from tqdm import tqdm
import cv2

face_cascade = cv2.CascadeClassifier('face.xml')
profiles = []
width_height = 100
len_before = 0
np.set_printoptions(suppress=True)


class ANN:
    def __init__(self):
        try:
            self.weights1 = np.load('Profiles/{0}/weights1.npy'.format(listdir('Profiles/')[-1]))
            self.weights2 = np.load('Profiles/{0}/weights2.npy'.format(listdir('Profiles/')[-1]))
        except FileNotFoundError:
            self.weights1 = np.random.uniform(low=.0001, high=.9999, size=(4200, 200))
            self.weights2 = np.random.uniform(low=.0001, high=.9999, size=(200, 2))
        self.list = []

    @staticmethod
    def s(xx, deriv=False):
        if deriv:
            return 1/(1 + np.exp(-xx))*(1-1/(1 + np.exp(-xx)))
        else:
            return 1/(1 + np.exp(-xx))

    def train(self, target_val):
        target_output = None
        if target_val == 0:
            target_output = np.array([1., 0.])
        elif target_val == 1:
            target_output = np.array([0., 1.])
        for ii in range(300):
            inputs = np.asarray(self.list)
            hidden = self.s(np.dot(inputs, self.weights1))
            output = self.s(np.dot(hidden, self.weights2))
            output_error = target_output - output
            output_delta = output_error * self.s(output, deriv=True)
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.s(hidden, deriv=True)
            self.weights2 += hidden.T.dot(output_delta)
            self.weights1 += inputs.T.dot(hidden_delta)


def new_profile(new_prof_name):
    global cap, img
    cap = cv2.VideoCapture(0)
    try:
        mkdir('Profiles/{0}'.format(new_prof_name))
        for iiii in tqdm(range(150)):
            ret1, img1 = cap.read()
            cv2.imshow('Facial Recognition', img1)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            faces1 = face_cascade.detectMultiScale(img1, 1.3, 5)
            for (x1, y1, w1, h1) in faces1:
                cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                roi1 = gray1[y1:y1 + h1, x1:x1 + w1]
                Image.fromarray(roi1).resize((100, 100)).crop((20, 20, 80, 90)).save('Profiles/{0}/{1}.png'.format(new_prof_name, iiii))
            if len(listdir('Profiles/{0}'.format(new_prof_name))) != iiii:
                continue
            k1 = cv2.waitKey(30) & 0xff
            if k1 == 27:
                break
    except FileExistsError:
        path = input('Add to existing? ')
        if path in ['n', 'N']:
            pass
        else:
            try:
                profiles[-1][1].weights1 = np.load('Profiles/{0}/weights1.npy'.format(new_prof_name))
                profiles[-1][1].weights2 = np.load('Profiles/{0}/weights2.npy'.format(new_prof_name))
            except FileNotFoundError:
                pass
            for iiii in range(50):
                ret1, img1 = cap.read()
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
                for (x1, y1, w1, h1) in faces1:
                    cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    roi1 = gray1[y1:y1 + h1, x1:x1 + w1]
                    Image.fromarray(roi1).resize((width_height, width_height)).crop((20, 20, 80, 90)).save('Profiles/{0}/{1}.png'.format(new_prof_name, iiii+len(listdir('Profiles/{0}/'.format(new_prof_name)))))
                    cv2.imshow('Facial Recognition', img1)
                k1 = cv2.waitKey(30) & 0xff
                if k1 == 27:
                    break
    val1 = 0
    val2 = 0
    dataset1 = listdir('Profiles/{0}'.format(new_prof_name))
    dataset2 = listdir('Profiles/Other')
    to_pop = []
    for ii in range(len(dataset1)):
        dataset1[ii] = 'Profiles/{0}/{1}'.format(new_prof_name, dataset1[ii])
    for ii in range(len(dataset1)):
        if '.png' not in dataset1[ii]:
            to_pop.append(ii)
    print(to_pop)
    print(len(dataset1))
    try:
        dataset1.pop(to_pop[0]-1)
        dataset1.pop(to_pop[1]-2)
    except IndexError:
        pass
    to_pop = []
    for ii in range(len(dataset2)):
        if '.png' not in dataset2[ii]:
            to_pop.append(ii)
    try:
        dataset2.pop(to_pop[0]-1)
        dataset2.pop(to_pop[1]-2)
    except IndexError:
        pass
    for ii in range(len(dataset2)):
        dataset2[ii] = 'Profiles/Other/{0}'.format(dataset2[ii])
    val1_max = len(dataset1)
    val2_max = (len(dataset2) + val1_max) - len(dataset2)
    for _ in tqdm(range(val1_max+val2_max)):
        target = randint(0, 1)
        if val1 == val1_max-1:
            target = 1
        elif val2 == val2_max-1:
            target = 0
        if target == 0:
            val1 += 1
            img = np.asarray(cv2.Canny(np.asarray(Image.open(dataset1[val1])), 150, 200)).flatten()
            profiles[-1][1].list = np.asarray([img])
            profiles[-1][1].train(0)
        elif target == 1:
            val2 += 1
            frame = cv2.imread(dataset2[val2], 0)
            for (xi, yi, wi, hi) in face_cascade.detectMultiScale(frame, 1.3, 5):
                roi1 = frame[yi:yi + hi, xi:xi + wi]
                img = np.asarray(cv2.Canny(np.asarray(Image.fromarray(roi1).resize((width_height, width_height)).crop((20, 20, 80, 90))), 150, 200)).flatten()
                profiles[-1][1].list = np.asarray([img])
                profiles[-1][1].train(1)
    np.save('Profiles/{0}/weights1.npy'.format(profiles[-1][0]), profiles[-1][1].weights1)
    np.save('Profiles/{0}/weights2.npy'.format(profiles[-1][0]), profiles[-1][1].weights2)


iteration = 0


def read(inputs):
    global iteration
    hidden = ANN().s(np.dot(np.asarray(inputs), profiles[-1][1].weights1))
    output = ANN().s(np.dot(hidden, profiles[-1][1].weights2))
    if output[0] >= output[1]:
        return profiles[-1][0]
    else:
        return 'Other'


for x in range(len(listdir('Profiles/'))):
    if listdir('Profiles/')[x] != 'Other':
        profiles.append([listdir('Profiles/')[x], ANN()])
        try:
            profiles[-1][1].weights1 = np.load('Profiles/{0}/weights1.npy'.format(listdir('Profiles/')[x]))
            profiles[-1][1].weights2 = np.load('Profiles/{0}/weights2.npy'.format(listdir('Profiles/')[x]))
        except FileNotFoundError:
            new_profile(input('No users found. New user name: '))
    elif len(listdir('Profiles')) == 1:
        new_profile(input('No users found. New user name: '))

cap = cv2.VideoCapture(0)
bright = 0
no_face = 0
enhancer = None
iteration1 = 0
iteration2 = 0
iteration3 = 0

while True:
    ok, img = cap.read()
    if bright == 1:
        enhancer = ImageEnhance.Brightness(Image.fromarray(img))
        img = enhancer.enhance(.7)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2)
    if bright == 2:
        enhancer = ImageEnhance.Brightness(Image.fromarray(img))
        img = enhancer.enhance(2)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2)
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    name = ''
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        name = read(np.asarray(cv2.Canny(np.asarray(Image.fromarray(roi_gray).resize((width_height, width_height)).crop((20, 20, 80, 90))), 150, 200)).flatten())
        if faces == ():
            break
        if name == 'Other' and bright == 0:
            img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), (0, 0, 255), 2)
            iteration2 += 1
            if iteration2 >= 15:
                bright = 1
                iteration2 = 0
        elif name == 'Other' and bright == 1:
            img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), (0, 0, 255), 2)
            iteration2 += 1
            if iteration2 >= 15:
                bright = 2
                iteration2 = 0
        elif name == 'Other' and bright == 2:
            img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), (0, 0, 255), 2)
            iteration2 += 1
            if iteration2 >= 15:
                bright = 0
                iteration2 = 0
        elif name != 'Other':
            img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), (255, 0, 0), 2)
            iteration2 = 0
    if name != '':
        img = cv2.putText(np.asarray(img), name, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4, cv2.LINE_AA)
        iteration3 += 1
        if iteration3 == 10:
            iteration3 = 0
            iteration2 = 0
    else:
        iteration1 += 1
    if bright == 0 and name == '':
        if iteration1 >= 15:
            bright = 1
            iteration1 = 0
    elif bright == 1 and name == '':
        iteration1 += 1
        if iteration1 >= 15:
            bright = 2
            iteration1 = 0
    elif bright == 2 and name == '':
        iteration1 += 1
        if iteration1 >= 15:
            bright = 0
            iteration1 = 0
    cv2.imshow('Facial Recognition', np.asarray(img))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        quit()
