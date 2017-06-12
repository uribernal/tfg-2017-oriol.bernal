import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


movies = ['After_The_Rain', 'Attitude_Matters', 'Barely_legal_stories', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Cloudland', 'Damaged_Kung_Fu', 'Decay', 'Elephant_s_Dream', 'First_Bite', 'Full_Service', 'Islands', 'Lesson_Learned', 'Norm', 'Nuclear_Family', 'On_time', 'Origami', 'Parafundit', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', 'Superhero', 'Tears_of_Steel', 'The_room_of_franz_kafka', 'The_secret_number', 'To_Claire_From_Sonny', 'Wanted', 'You_Again']

path = 'C:/Users/Uri/Desktop/training_feat.h5'
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

fig = plt.figure(0)
plt.bar(np.arange(len(movies)), num_scenes)
plt.bar(np.arange(len(movies)), num_fear_scenes)
plt.show()
fig = plt.figure(1)
plt.bar(np.arange(len(movies)), np.array(num_fear_scenes)/np.array(num_scenes))
plt.show()
