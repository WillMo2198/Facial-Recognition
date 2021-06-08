from PIL import Image, ImageEnhance
from random import randint
from tqdm import trange
import numpy as np
import pickle
import cv2
import os


face_cascade = cv2.CascadeClassifier('face.xml')
np.set_printoptions(suppress=False)
cap = cv2.VideoCapture(0)


def nonlin(xx, deriv=False):
    if deriv:
        return nonlin(xx)*(1-nonlin(xx))
    return 1/(1 + np.exp(-xx))


def new_user(name):
    try:
        os.mkdir('Users/{0}'.format(name))
        layer1 = np.load('Users/{0}/weights1.npy'.format(name))
        layer2 = np.load('Users/{0}/weights2.npy'.format(name))
    except FileExistsError or FileNotFoundError:
        layer1 = np.random.uniform(low=0., high=1., size=(75 * 75, 25))
        layer2 = np.random.uniform(low=0., high=1., size=(25, 1))
    iteration2 = 0
    iteration3 = 0
    np.random.seed(1)
    in_data = []
    i = 0
    if len(os.listdir('Users/{0}'.format(name))) < 3:
        while len(os.listdir('Users/{0}'.format(name))) < 102:
            _, img1 = cap.read()
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
            if len(faces1) > 0:
                for (x1, y1, w1, h1) in faces1:
                    roi1 = Image.fromarray(img1[y1:y1 + h1, x1:x1 + w1])
                    enhancer1 = ImageEnhance.Brightness(roi1)
                    roi1 = enhancer1.enhance(1.5)
                    roi1 = roi1.resize((75, 75))
                    roi1 = cv2.Canny(np.asarray(roi1), 100, 400)
                    cv2.imshow('img', img1)
                    Image.fromarray(roi1).save('Users/{0}/{1}.jpg'.format(name, i))
                    k1 = cv2.waitKey(30) & 0xff
                    i += 1
                    if k1 == 27:
                        cap.release()
                        cv2.destroyAllWindows()
            else:
                continue
    else:
        pass
    out = randint(0, 1)
    for _ in trange(200):
        if out == 0 and iteration2 == 100:
            out = 1
        if out == 1 and iteration3 == len(os.listdir('Users/{0}'.format(name))):
            out = 0
        if out == 0:
            img1 = cv2.imread('Other/{0}'.format(os.listdir('Other')[iteration2]), 0)
            iteration2 += 1
            faces1 = face_cascade.detectMultiScale(img1, 1.3, 5)
            if len(faces1) == 0:
                continue
            for (x1, y1, w1, h1) in faces1:
                roi_color1 = Image.fromarray(img1[y1:y1 + h1, x1:x1 + w1])
                enhancer1 = ImageEnhance.Brightness(roi_color1)
                roi_color1 = enhancer1.enhance(1.5)
                roi_color1 = roi_color1.resize((75, 75))
                roi_color1 = cv2.Canny(np.asarray(roi_color1), 100, 400)
                in_data = np.asarray(Image.fromarray(roi_color1).resize((75, 75)))
        else:
            in_data = cv2.imread('Users/{0}/{1}'.format(name, os.listdir('Users/{0}/'.format(name))[iteration3]), 0)
            iteration3 += 1
        try:
            in_data = in_data.flatten()
            if not np.isnan(in_data[0]):
                for iii in range(75):
                    l0 = np.asarray([in_data])
                    l1 = nonlin(np.dot(l0, layer1))
                    l2 = nonlin(np.dot(l1, layer2))
                    l2_error = out - l2
                    l2_delta = l2_error * nonlin(l2, deriv=True)
                    l1_error = l2_delta.dot(layer2.T)
                    l1_delta = l1_error * nonlin(l1, deriv=True)
                    layer2 += l1.T.dot(l2_delta)
                    layer1 += l0.T.dot(l1_delta)
        except AttributeError or ValueError:
            continue
        out = randint(0, 1)
    np.save('Users/{0}/weights1.npy'.format(name), layer1)
    np.save('Users/{0}/weights2.npy'.format(name), layer2)
    contendors = os.listdir('Users')
    contendors.remove(name)
    print(contendors)
    iteration = 0
    if len(contendors) != 0:
        for i in range(len(contendors)):
            player_dir = os.listdir('Users/{0}'.format(name))
            opp_dir = os.listdir('Users/{0}'.format(contendors[iteration]))
            iteration2 = 0
            iteration3 = 0
            for _ in trange(int(len(player_dir)+len(opp_dir)/2)):
                out = randint(0, 1)
                if out == 0:
                    in_data = cv2.imread('Users/{0}/{1}'.format(contendors[iteration], opp_dir[iteration2]), 0).flatten()
                elif out == 1:
                    in_data = cv2.imread('Users/{0}/{1}'.format(name, player_dir[iteration3]), 0).flatten()
                for iiii in range(50):
                    l0 = np.asarray([in_data])
                    l1 = nonlin(np.dot(l0, layer1))
                    l2 = nonlin(np.dot(l1, layer2))
                    l2_error = out - l2
                    l2_delta = l2_error * nonlin(l2, deriv=True)
                    l1_error = l2_delta.dot(layer2.T)
                    l1_delta = l1_error * nonlin(l1, deriv=True)
                    layer2 += l1.T.dot(l2_delta)
                    layer1 += l0.T.dot(l1_delta)
        np.save('Users/{0}/weights1.npy'.format(name), layer1)
        np.save('Users/{0}/weights2.npy'.format(name), layer2)
        iteration += 1


try:
    user_color = pickle.load(open("user-colors.pkl", "rb"))

except FileNotFoundError:
    user_color = {}
    for iiiii in range(len(os.listdir('Users'))+1):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        if iiiii == len(os.listdir('Users')):
            user_color['Other'] = color
        else:
            user_color[os.listdir('Users')[iiiii]] = color
    pickle.dump(user_color, open("user-colors.pkl", "wb"))

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = Image.fromarray(img[y:y + h, x:x + w])
        enhancer = ImageEnhance.Brightness(roi)
        roi = enhancer.enhance(1.5)
        roi = roi.resize((75, 75))
        roi = cv2.Canny(np.asarray(roi), 100, 400)
        cv2.imshow('lines', np.asarray(roi))
        roi = np.asarray(roi).flatten()
        input_layer = np.asarray(roi)
        user_out = []
        out1 = []
        user = {}
        user_name = ''
        if len(os.listdir('Users')) > 1:
            for ii in os.listdir('Users'):
                hidden = nonlin(np.dot(input_layer, np.load('Users/{0}/weights1.npy'.format(ii))))
                output = nonlin(np.dot(hidden, np.load('Users/{0}/weights2.npy'.format(ii))))
                user[ii] = output[0]
            if user[max(user, key=user.get)] <= .5:
                user_name = 'Other'
                img = cv2.rectangle(img, (x, y), (x + w, y + h), user_color['Other'], 2)
                cv2.imshow('img', np.asarray(img))
            else:
                user_name = max(user, key=user.get)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), user_color[user_name], 2)
                cv2.imshow('img', np.asarray(img))
            print(user_name)
        else:
            hidden = nonlin(np.dot(input_layer, np.load('Users/{0}/weights1.npy'.format(os.listdir('Users')[0]))))
            output = nonlin(np.dot(hidden, np.load('Users/{0}/weights2.npy'.format(os.listdir('Users')[0]))))
            print(output)
            if output > 0.5:
                user_name = os.listdir('Users')[0]
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('img', np.asarray(img))
            elif output <= 0.5:
                user_name = 'Other'
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('img', np.asarray(img))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
