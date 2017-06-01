import h5py
import os.path
#from fine_tuning.Audio import vgg19
from keras.applications.vgg19 import VGG19
from helper import DatasetManager as Dm
#from fine_tuning.Audio.vgg19 import VGG19
from keras.layers import Input
from helper import AudioHelper as Ah
import numpy as np


path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/'
name = 'emotional_impact'
extension = '.h5'
path_file = path + name + extension

win_frames = 23520
num_stft = 96
num_filter_banks = 64

if not os.path.isfile(path_file):
    # Create the HDF5 file
    hdf = h5py.File(path_file, 'w')
    hdf.close()

movies = Dm.get_movies_names()

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(3, 96, 64))  # this assumes K.image_data_format() == 'channels_last'

#model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, input_shape=None)
model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, input_shape=None)

for movie in movies:
    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5', 'r') as hdf:
        acoustic_data = np.array(hdf.get('dev/acoustic_data/' + movie))
    #print(acoustic_data.shape)
    #res = acoustic_data.shape[0] % 3
    #cte = acoustic_data.shape[0] - res
    #acoustic_data = acoustic_data[:cte, :, :]
    cte = acoustic_data.shape[0]
    a = np.append(acoustic_data, acoustic_data)
    acoustic_data = np.append(a, acoustic_data)
    computed_audio = acoustic_data.reshape(3, cte, 96, 64)
    #print(computed_audio.shape)
    computed_audio = computed_audio.transpose(1, 0, 2, 3)
    print(computed_audio.shape)


    predictions = model.predict(computed_audio[:, :, :, :], batch_size=1)
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[1]*predictions.shape[2]*predictions.shape[3])
    print(predictions.shape)
    with h5py.File(path_file, 'r+') as hdf:
        hdf.create_dataset('dev/acoustic_features/' + movie, data=predictions, compression='gzip', compression_opts=9)
    print('{} stored'.format(movie))
print('Finished')