import h5py
import os.path
from fine_tuning.Audio import vgg19
from helper import DatasetManager as Dm

path = '/home/uribernal/Desktop/MediaEval2017/data/data/'
name = 'emotional_impact'
extension = '.h5'
path_file = path + name + extension

if not os.path.isfile(path_file):
    # Create the HDF5 file
    hdf = h5py.File(path_file, 'w')
    hdf.close()

movies = Dm.get_movies_names()
labels_type = ['valence', 'arousal', 'fear']

time_prediction_constant = 10
time_shift_prediction_constant = 5
input_size = (112, 112)
num_frames = 16
num_visual_feat = 4096  # Features after C3D convolutions (visual feature extractor model)
num_acoustic_feat = 4096  # Features after CD convolutions (acoustic feature extractor model)
win_frames = 44100
num_stft = 98
num_filter_banks = 64

audio_features = vgg19.get_features_from_vgg19(movies)
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('audio_features', data=audio_features, compression='gzip', compression_opts=9)
print('Audio features Stored:')

