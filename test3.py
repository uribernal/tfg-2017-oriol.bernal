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


movies = Dm.get_movies_names()
path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'
for movie in movies:
    #movie = 'Decay'
    with h5py.File(path, 'r') as hdf:
        labels = np.array(hdf.get('dev/ground_truth_data/' + movie))

    print(labels.shape)

    new_array = modify_vector2(labels[:, 0])
    print(new_array.shape)
    new_array2 = modify_vector2(labels[:, 1])

    new_array3 = modify_vector2(labels[:, 2])

    caca = np.append(new_array, new_array2)
    caca = caca.reshape(2, new_array.shape[0])

    caca = np.append(caca, new_array3)
    caca = caca.reshape(3, new_array.shape[0])
    caca = caca.transpose(1, 0)

    print(caca.shape)
    #break

    with h5py.File(path, 'r+') as hdf:
        hdf.create_dataset('dev/modified_labels/' + movie, data=caca, compression='gzip', compression_opts=9)
    print('{} stored'.format(movie))
