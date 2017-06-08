import h5py
import os.path
from helper import DatasetManager as Dm


path = '/home/uribernal/Desktop/MediaEval2017/data/data/'
name = 'emotional_impact'
extension = '.h5'
path_file = path + name + extension

if not os.path.isfile(path_file):
    # Create the HDF5 file
    hdf = h5py.File(path_file, 'w')
    hdf.close()


time_prediction_constant = 10
time_shift_prediction_constant = 5
input_size = (112, 112)
num_frames = 16
num_visual_feat = 4096  # Features after C3D convolutions (visual feature extractor model)
num_acoustic_feat = 4096  # Features after CD convolutions (acoustic feature extractor model)
win_frames = 44100
num_stft = 98
num_filter_banks = 64
'''
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('time_prediction_constant', data=time_prediction_constant, compression='gzip', compression_opts=9)
    hdf.create_dataset('time_shift_prediction_constant', data=time_shift_prediction_constant, compression='gzip', compression_opts=9)
    hdf.create_dataset('input_size', data=input_size, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_frames', data=num_frames, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_visual_feat', data=num_visual_feat, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_acoustic_feat', data=num_acoustic_feat, compression='gzip', compression_opts=9)
    hdf.create_dataset('win_frames', data=win_frames, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_stft', data=num_stft, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_filter_banks', data=num_filter_banks, compression='gzip', compression_opts=9)
print('Constants Stored:')
'''

movies = Dm.get_movies_names()
labels_type = ['valence', 'arousal', 'fear']


labels = Dm.get_ground_truth_data(movies)
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
print('Labels Stored:')

frames, fps, duration = Dm.get_videos_info(movies)
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('frames', data=frames, compression='gzip', compression_opts=9)
    hdf.create_dataset('fps', data=fps, compression='gzip', compression_opts=9)
    hdf.create_dataset('duration', data=duration, compression='gzip', compression_opts=9)
print('Videos info Stored:')

fs, num_audio_samples = Dm.get_audios_info(movies)
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('fs', data=fs, compression='gzip', compression_opts=9)
    hdf.create_dataset('num_audio_samples', data=num_audio_samples, compression='gzip', compression_opts=9)
print('Audios info Stored:')

predictions_length = Dm.get_predictions_length(movies)  # The number of predictions per video (depend on its length)
with h5py.File(path_file, 'r+') as hdf:
    hdf.create_dataset('predictions_length', data=predictions_length, compression='gzip', compression_opts=9)
print('Predictions length Stored:')

