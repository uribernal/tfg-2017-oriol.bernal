from helper import AudioHelper as Ah
from helper import DatasetManager as Dm
from helper import VideoHelper as Vh

movies = Dm.get_movies_names()

# Extract audios
#Ah.extract_audios()

# Change FPS in videos to 30
path_input = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/'
path_output = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/30fps/'
ext = '.mp4'



import subprocess
for movie in movies:
    video_input = path_input+movie+ext
    video_output = path_output+movie+ext

    c = 'ffmpeg -y -i ' + video_input + ' -r 30 -s 112x112 -c:v libx264 -b:v 3M -strict -2 -movflags faststart '+video_output
    subprocess.call(c, shell=True)

# Change FS in audios to 44100
#############################################################################################################
import h5py
import numpy as np
input_size = (112, 112)
length = 16
for movie in movies:
    with h5py.File('/home/uribernal/Desktop/MediaEval2017/data/visual_data.h5', 'r') as hdf:
        labels = np.array(hdf.get('labels/' + movie))

    input_video = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
                  'LIRIS-ACCEDE-continuous-movies/30fps/' + movie + '.mp4'

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

    path_file = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'
    import os
    print('Saving film...')
    if not os.path.isfile(path_file):
        # Create the HDF5 file
        hdf = h5py.File(path_file, 'w')
        hdf.close()
    with h5py.File(path_file, 'r+') as hdf:
        hdf.create_dataset('dev/visual_data/' + movie, data=video_array, compression='gzip', compression_opts=9)
