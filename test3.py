import h5py
import os.path
import numpy as np
from fine_tuning.Audio import vgg19
from helper import DatasetManager as Dm
from fine_tuning.Audio.vgg19 import VGG19
from keras.layers import Input
from helper import AudioHelper as Ah
from fine_tuning.Audio import vgg19


def modify_vector(labels):
    cte1 = np.ones(10)
    cte2 = np.ones(9)
    cte3 = np.ones(8)
    new_array = cte1 * labels[0]
    for i in range(labels.shape[0] - 1):
        if (i % 2 == 0):
            cte = cte1
        elif (i % 5 == 0):
            cte = cte3
        else:
            cte = cte2
        a = (cte * labels[i] + cte * labels[i + 1]) / 2.0
        new_array = np.append(new_array, a)
    a = cte1 * labels[-1]
    new_array = np.append(new_array, a)
    return new_array


def modify_vector2(labels):
    a1 = np.ones(9)
    a2 = np.ones(10)
    new_array = a1 * labels[0]

    sequence = [a1, a2, a1, a1, a2, a1, a2, a1]
    j = 1
    for i in range(labels.shape[0] - 1):
        if j == 8:
            j = 0
        cte = sequence[j]
        a = (cte * labels[i] + cte * labels[i + 1]) / 2.0
        new_array = np.append(new_array, a)
        j += 1
    if j == 8:
        j = 0
    a = sequence[j] * labels[-1]
    new_array = np.append(new_array, a)
    return new_array


def demodify_vector(labels):
    a1 = 9
    a2 = 10

    sequence = [a1, a2, a1, a1, a2, a1, a2, a1]

    cont = True
    cte = a1
    new_array = [labels[0]]
    i = cte
    j = 1
    while cont:
        rest = labels.shape[0] - i
        if j == 8:
            j = 0
        cte = sequence[j]
        #print('{}:{}'.format(i, i+cte))
        #a = np.sum(labels[i:(i+cte)]) / cte
        a = 2*labels[i]-new_array[-1]
        i += cte
        if rest <= 29:
            cont = False
        new_array = np.append(new_array, a)
        j += 1
    a = labels[-1]
    new_array = np.append(new_array, a)
    return new_array


def calculate_PCC(y_pred, y_true):
    m1 = np.mean(y_pred)
    m2 = np.mean(y_true)
    y_pred_norm = y_pred -m1
    y_true_norm = y_true - m2
    nom = np.sum(y_pred_norm*y_true_norm)
    den = np.sqrt(np.sum(y_pred_norm**2))*np.sqrt(np.sum(y_true_norm**2))

    return nom/den



val = np.array([0.02702231, 0.17112755, 0.18911039, 0.17959486, 0.17049178, 0.17258332, 0.20127973, 0.1854717, 0.18314009, 0.17307961])
ar = np.array([0.02418533, 0.16093774, 0.15094463, 0.15372311, 0.15803463, 0.15863948, 0.14824309, 0.16629058, 0.15309987, 0.15245098])


