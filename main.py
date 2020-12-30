from PIL import Image, ImageEnhance, UnidentifiedImageError
from warnings import filterwarnings
from os import listdir, mkdir
from random import randint
from tqdm import tqdm
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('face.xml')
profiles = []
np.set_printoptions(suppress=True)
filterwarnings("ignore")


class ANN:
    def __init__(self):
        try:
            self.weights1 = np.load('Profiles/{0}/weights1.npy'.format(listdir('Profiles/')[-1]))
            self.weights2 = np.load('Profiles/{0}/weights2.npy'.format(listdir('Profiles/')[-1]))
        except FileNotFoundError:
            self.weights1 = np.random.uniform(low=.0001, high=.9999, size=(4200, 200))
            self.weights2 = np.random.uniform(low=.0001, high=.9999, size=(200, 2))
        self.list = []

    def train(self, target_val):
        target_output = None
        if target_val == 0:
            target_output = np.array([1., 0.])
        elif target_val == 1:
            target_output = np.array([0., 1.])
        for ii in range(50):
            inputs = np.asarray(self.list)
            hidden = s(np.dot(inputs, self.weights1))
            output = s(np.dot(hidden, self.weights2))
            output_error = target_output - output
            output_delta = output_error * s(output, deriv=True)
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * s(hidden, deriv=True)
            self.weights2 += hidden.T.dot(output_delta)
            self.weights1 += inputs.T.dot(hidden_delta)


def s(xx, deriv=False):
    if deriv:
        return 1/(1 + np.exp(-xx))*(1-1/(1 + np.exp(-xx)))
    else:
        return 1/(1 + np.exp(-xx))


def new_profile(new_prof_name):
    global cap, img
    cap = cv2.VideoCapture(0)
    try:
        mkdir('Profiles/{0}'.format(new_prof_name))
        loop1 = 0
        while True:
            ret1, img1 = cap.read()
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            faces1 = face_cascade.detectMultiScale(img1, 1.3, 5)
            for (x1, y1, w1, h1) in faces1:
                cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                roi1 = gray1[y1:y1 + h1, x1:x1 + w1]
                Image.fromarray(roi1).resize((100, 100)).crop((20, 20, 80, 90)).save('Profiles/{0}/{1}.png'.format(new_prof_name, loop1))
            cv2.imshow('Facial Recognition', img1)
            cv2.waitKey(1)
            if len(faces1) != 0:
                loop1 += 1
            if loop1 == 500:
                return
    except FileExistsError:
        path = input('Add more images? ')
        if path in ['n', 'N']:
            pass
        else:
            loop1 = 0
            while True:
                ret1, img1 = cap.read()
                gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                faces1 = face_cascade.detectMultiScale(img1, 1.3, 5)
                for (x1, y1, w1, h1) in faces1:
                    cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
                    roi1 = gray1[y1:y1 + h1, x1:x1 + w1]
                    Image.fromarray(roi1).resize((100, 100)).crop((20, 20, 80, 90)).save('Profiles/{0}/{1}.png'.format(new_prof_name, len('Profiles/{0}'.format(new_prof_name))+1))
                cv2.imshow('Facial Recognition', img1)
                cv2.waitKey(1)
                if len(faces1) != 0:
                    loop1 += 1
                if len(listdir('Profiles/{0}'.format(new_prof_name))) == 500:
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
    val_max = len(dataset1)
    for _ in tqdm(range(val_max+val_max)):
        target = randint(0, 1)
        if val1 == val_max-1:
            target = 1
        elif val2 == val_max-1:
            target = 0
        if target == 0:
            val1 += 1
            try:
                img = np.asarray(cv2.Canny(np.asarray(Image.open(dataset1[val1])), 150, 200))
                profiles[-1][1].list = np.asarray([img.flatten()])
                profiles[-1][1].train(0)
            except UnidentifiedImageError:
                pass
        elif target == 1:
            val2 += 1
            try:
                frame = cv2.imread(dataset2[val2], 0)
                for (xi, yi, wi, hi) in face_cascade.detectMultiScale(frame, 1.3, 5):
                    roi1 = frame[yi:yi + hi, xi:xi + wi]
                    img = np.asarray(cv2.Canny(np.asarray(Image.fromarray(roi1).resize((100, 100)).crop((20, 20, 80, 90))), 150, 200))
                    profiles[-1][1].list = np.asarray([img.flatten()])
                    profiles[-1][1].train(1)
            except UnidentifiedImageError:
                pass
    np.save('Profiles/{0}/weights1.npy'.format(profiles[-1][0]), profiles[-1][1].weights1)
    np.save('Profiles/{0}/weights2.npy'.format(profiles[-1][0]), profiles[-1][1].weights2)


iteration = 0


def read(inputs, user):
    global iteration
    hidden = s(np.dot(np.asarray(inputs), profiles[user][1].weights1))
    output = s(np.dot(hidden, profiles[user][1].weights2))
    if output[0] >= output[1]:
        return profiles[user][0]
    else:
        return 'Other'


for x in range(len(listdir('Profiles/'))):
    if listdir('Profiles/')[x] != 'Other':
        profiles.append([listdir('Profiles/')[x], ANN(), [], 0, (randint(0, 255), randint(0, 255), randint(0, 255))])
        edit = input('Edit Users? ')
        if edit in ['y', 'Y']:
            new_profile(input('Name: '))
        else:
            break
    elif len(listdir('Profiles')) == 1:
        new_profile(input('No users found. New user name: '))
        new_profile(listdir('Profiles/')[-1])
        continue

cap = cv2.VideoCapture(0)
bright = 0
enhancer = None
iteration1 = 0
iteration2 = 0
iteration3 = 0
face = 0
no_face = 0
avg = 0
avg1 = 0
loop = 1
other_color = (0, 0, 0)

while True:
    ok, img = cap.read()
    if bright == 1:
        other_color = (255, 255, 255)
        try:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2)
        except AttributeError:
            enhancer = ImageEnhance.Brightness(Image.fromarray(img))
            img = enhancer.enhance(.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2)
    elif bright == 2:
        other_color = (255, 255, 255)
        try:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(3)
        except AttributeError:
            enhancer = ImageEnhance.Brightness(Image.fromarray(img))
            img = enhancer.enhance(3)
    try:
        gray = cv2.cvtColor(Image.fromarray(img), cv2.COLOR_RGB2GRAY)
    except TypeError:
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    except AttributeError:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    name = ''
    edges = []
    for i in range(len(profiles)):
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), profiles[i][4], 2)
            edges = np.asarray(cv2.Canny(np.asarray(Image.fromarray(roi_gray).resize((100, 100)).crop((20, 20, 80, 90))), 150, 200))
            cv2.imshow('Edges', edges)
            name = read(edges.flatten(), i-1)
            if faces == ():
                break
            if name == 'Other':
                img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), other_color, 2)
                if bright == 0:
                    iteration2 += 1
                    if iteration2 >= 20:
                        bright = 1
                        iteration2 = 0
                elif bright == 1:
                    iteration2 += 1
                    if iteration2 >= 20:
                        bright = 2
                        iteration2 = 0
                elif bright == 2:
                    iteration2 += 1
                    if iteration2 >= 20:
                        bright = 0
                        iteration2 = 0
            else:
                img = cv2.rectangle(np.asarray(img), (x, y), (x + w, y + h), profiles[i][4], 2)
                iteration3 += 1
                face += 1
                avg = int((face / loop) * 100)
                if iteration3 >= 10:
                    iteration1 = 0
                    iteration2 = 0
                    iteration3 = 0
        img = cv2.putText(np.asarray(img), 'Other: {0}%'.format(100-avg), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, other_color, 4, cv2.LINE_AA)
        img = cv2.putText(np.asarray(img), '{0}: {1}%'.format(profiles[i][0], avg), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, profiles[i][4], 4, cv2.LINE_AA)
        loop += 1
        if loop == 100:
            loop = 1
            face = 0
            no_face = 0
            avg = 0
            avg1 = 0
        if name == '':
            if bright == 0:
                iteration1 += 1
                if iteration1 >= 25:
                    bright = 1
                    iteration1 = 0
                    iteration2 = 0
                    iteration3 = 0
                    profiles[i][2] = [0]
            elif bright == 1:
                iteration1 += 1
                if iteration1 >= 25:
                    bright = 2
                    iteration1 = 0
                    iteration2 = 0
                    iteration3 = 0
                    profiles[i][2] = [0]
            elif bright == 2:
                iteration1 += 1
                if iteration1 >= 25:
                    bright = 0
                    iteration1 = 0
                    iteration2 = 0
                    iteration3 = 0
                    profiles[i][2] = [0]
        cv2.imshow('Facial Recognition', np.asarray(img))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()
