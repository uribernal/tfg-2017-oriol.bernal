import h5py
import os.path
from helper import AudioHelper as Ah
from helper import DatasetManager as Dm
import numpy as np


path = '/home/uribernal/Desktop/MediaEval2017/data/data/'
name = 'acoustic_data_final'
extension = '.h5'
path_file = path + name + extension

if not os.path.isfile(path_file):
    # Create the HDF5 file
    hdf = h5py.File(path_file, 'w')
    hdf.close()

movies = Dm.get_movies_names()
labels_type = ['valence', 'arousal', 'fear']

win_frames = 23520
num_stft = 96
num_filter_banks = 64

movies = Dm.get_movies_names()
predictions_length = Dm.get_predictions_length(movies)

for cont, movie in enumerate(movies):
    resized_audio = Ah.get_resized_audio2(cont, win_frames=win_frames, print_info=False)
    computed_audio = Ah.compute_STFT_and_MelBank(resized_audio, print_info=False)
    print(computed_audio.shape)
    with h5py.File(path_file, 'r+') as hdf:
        hdf.create_dataset('acoustic_data/'+movie, data=computed_audio, compression='gzip', compression_opts=9)
    print('Acoustic data length Stored:')

    #computed_audio = computed_audio.reshape(int(computed_audio.shape[0] / 5), 5, 98, 64)
#with h5py.File(path_file, 'r+') as hdf:
    #hdf.create_dataset('acoustic_data', data=acoustic_data, compression='gzip', compression_opts=9)
#print('Acoustic data length Stored:')
