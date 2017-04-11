import math

import h5py
import numpy as np

from continuousMovies.featureExtraction import VideoGenerator as vg

DATABASE_PATH = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/BaseDeDades.h5'

def create_hdf5(dim: tuple):
    """ Creates an HDF5 file for storing the dataset
    Args:
        dim (tuple): Tuple with the height and width of the videos
    """

    # Create the HDF5 file
    hdf = h5py.File(DATABASE_PATH, 'w')

    # Create the structure of the DB
    MOVIES = hdf.create_group('MOVIES')
    AUDIOS = hdf.create_group('AUDIOS')
    VALENCE = hdf.create_group('LABELS/VALENCE')
    AROUSAL = hdf.create_group('LABELS/AROUSAL')
    FEAR = hdf.create_group('LABELS/FEAR')

    # Set attributes
    MOVIES.attrs['HEIGHT'] = str(dim[0])
    MOVIES.attrs['WIDTH'] = str(dim[1])
    hdf.close()

def store_film(film_path: str, film_name: str, film_extension: str, fps=1):
    """ Stores the RGB frames of a movie
    Args:
        path (string): path where the file is stored
    """

    with h5py.File(DATABASE_PATH, 'r') as hdf:
        # Store movie
        MOVIES = hdf.get('MOVIES')

        # Get dimensions of the video
        d = list(MOVIES.attrs.values())
        dim = (int(d[0]), int(d[1]))

    # Read movie
    video_path = film_path+film_name+film_extension
    n_frames = vg.get_num_frames(video_path)
    duration = vg.get_duration(video_path)
    real_fps = n_frames/duration
    frame = 0,
    data = np.array([])
    for i in range(0, math.floor(duration)+1):
        a = vg.video_to_array(video_path, resize=dim, start_frame=i, length=1)
        data = np.append(data, a[:,0,:,:])

    data = data.reshape(i+1, 3, dim[0], dim[1])

    with h5py.File(DATABASE_PATH, 'r+') as hdf:
        # Store movie
        MOVIES = hdf.get('MOVIES')
        film = MOVIES.create_dataset(film_name, data=data, compression='gzip', compression_opts=9)
    print('Stored: '+film_name)


def get_film(film_name: str):
    """ Returns the RGB frames of a movie
    Args:
        film_name (string): name of the film to be retrieved
    """
    with h5py.File(DATABASE_PATH, 'r') as hdf:
        # Read movie
        MOVIES = hdf.get('MOVIES')

        data = MOVIES.get(film_name)
        movie = np.array(data)
        return movie

def get_labels(film_name: str):
    """ Returns the RGB frames of a movie
    Args:
        film_name (string): name of the film to be retrieved
    """
    with h5py.File(DATABASE_PATH, 'r') as hdf:
        # Read movie
        AROUSAL = hdf.get('LABELS/AROUSAL')
        VALENCE = hdf.get('LABELS/VALENCE')

        ar = AROUSAL.get(film_name)
        val = VALENCE.get(film_name)

        arousal = np.array(ar)
        valence = np.array(val)
        return arousal, valence



def save_labels_arousal(movie: str):
    path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-annotations/continuous-annotations/'
    extension = '_Arousal.txt'
    time = []
    mean = []
    std = []
    with open(path+movie+extension) as f:
        for line in f:
            l = line.strip().split()
            time.append(l[0])
            mean.append(l[1])
            std.append(l[2])
    mean.pop(0)
    data = np.array(mean, dtype=float)
    mean = []
    with h5py.File(DATABASE_PATH, 'r+') as hdf:
        AROUSAL = hdf.get('LABELS/AROUSAL')
        AROUSAL.create_dataset(movie, data=data, compression='gzip', compression_opts=9)

def save_labels_valence(movie: str):
    path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-annotations/continuous-annotations/'
    extension = '_Valence.txt'
    time = []
    mean = []
    std = []

    with open(path + movie + extension) as f:
        for line in f:
            l = line.split()
            time.append(l[0])
            mean.append(l[1])
            std.append(l[2])
    mean.pop(0)
    print('------>'+str(len(time)))
    print('------>'+str(len(mean)))
    print('------>'+str(len(std)))

    data = np.array(mean, dtype=float)
    print(data.shape)
    mean = []
    with h5py.File(DATABASE_PATH, 'r+') as hdf:
        VALENCE = hdf.get('LABELS/VALENCE/')
        VALENCE.create_dataset(movie, data=data, compression='gzip', compression_opts=9)


def get_dimensions():
    """ Returns the weght and width of the movies """

    with h5py.File(DATABASE_PATH, 'r') as hdf:
        # Read movie
        MOVIES = hdf.get('MOVIES')
        v = list(MOVIES.attrs.values())
        return (int(v[0]), int(v[1]))

def get_movies_names():
    mypath = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/'
    from os import listdir
    from os.path import isfile, join
    movies = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i, file in enumerate(movies):
        movies[i] = file[:-4]
    return(movies)


'''
from helper import bot
import time
bot.sendMessage("START")
start = time.time()

create_hdf5((224,224))
movies = get_movies_names()
for film in movies:
    store_film(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/',
        film, '.mp4')
end = time.time()
elapsed = end - start

for film in movies:
    print(film)
    save_labels_arousal(film)
    save_labels_valence(film)

bot.sendMessage("Elapsed time:" + str(elapsed))
print(get_dimensions())

'''
movies = get_movies_names()
for movie in movies:
    a = get_film(movie)
    b, c = get_labels(movie)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print('-------------')