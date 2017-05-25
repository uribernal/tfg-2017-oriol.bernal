from helper import DatasetManager as Dm
from helper import ModelGenerator as Mg
from helper import VideoHelper as Vh
import numpy as np
import h5py


movies = Dm.get_movies_names()
input_size = (112, 112)
length = 16
movies = movies[23:]
for movie in movies:
    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/visual_data.h5', 'r') as hdf:
        labels = np.array(hdf.get('labels/' + movie))

    input_video = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
                  'LIRIS-ACCEDE-continuous-movies/continuous-movies/' + movie + '.mp4'

    print('Reading Video...')
    video_array = Vh.video_to_array(input_video, resize=input_size)
    if video_array is None:
        raise Exception('The video could not be read')

    nb_frames = Vh.get_num_frames(input_video)
    duration = Vh.get_duration(input_video)
    fps = nb_frames / duration
    print('Duration: {:.1f}s'.format(duration))
    print('FPS: {:.1f}'.format(fps))
    print('Number of frames: {}'.format(nb_frames))

    nb_clips = nb_frames // length
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array[:nb_clips * length, :, :, :]
    video_array = video_array.reshape((nb_clips, length, 3, input_size[0], input_size[1]))
    video_array = video_array.transpose(0, 2, 1, 3, 4)

    path_file = '/home/uribernal/Desktop/MediaEval2017/data/data/raw_video_final.h5'
    import os
    print('Saving film...')
    if not os.path.isfile(path_file):
        # Create the HDF5 file
        hdf = h5py.File(path_file, 'w')
        hdf.close()
    with h5py.File(path_file, 'r+') as hdf:
        hdf.create_dataset('dev/' + movie, data=video_array, compression='gzip', compression_opts=9)
