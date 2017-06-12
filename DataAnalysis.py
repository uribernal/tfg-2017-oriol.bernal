import numpy as np
import h5py
import os
from helper import DatasetManager as Dm
import matplotlib.pyplot as plt


movies = Dm.get_movies_names()

path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/training_feat.h5'
if not os.path.isfile(path):
    # Create the HDF5 file
    hdf = h5py.File(path, 'w')
    hdf.close()

l = np.array([])
num_scenes = np.array([])
num_fear_scenes = np.array([])

for movie in movies:
    with h5py.File(path, 'r') as hdf:
        labels = np.array(hdf.get('dev/labels/' + movie))
    labels = labels.reshape(labels.shape[0], 1, labels.shape[1])
    num_scenes = np.append(num_scenes, labels.shape[0])
    num_fear_scenes = np.append(num_fear_scenes, np.count_nonzero(labels[:,:,2]))
    l = np.append(l, labels)

labels = l.reshape(l.shape[0] // 3, 1, 3)

plt.bar(np.arange(len(movies)), num_scenes)
plt.bar(np.arange(len(movies)), num_fear_scenes)
plt.show