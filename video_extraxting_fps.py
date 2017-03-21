import numpy as np
import h5py
from featureExtraction import VideoGenerator as vg
import math
DATABASE_PATH = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/dataset.h5'

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

    print(i)
    data = data.reshape(i+1, 3, dim[0], dim[1])
    print(data.shape)

    with h5py.File(DATABASE_PATH, 'r+') as hdf:
        # Store movie
        MOVIES = hdf.get('MOVIES')
        print(MOVIES)
        film = MOVIES.create_dataset(film_name, data=data, compression='gzip', compression_opts=9)


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


def get_dimensions():
    """ Returns the weght and width of the movies """

    with h5py.File(DATABASE_PATH, 'r') as hdf:
        # Read movie
        MOVIES = hdf.get('MOVIES')
        v = list(MOVIES.attrs.values())
        print(v)
        return (int(v[0]), int(v[1]))

def get_movies():
    mypath = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/'
    from os import listdir
    from os.path import isfile, join
    movies = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i, file in enumerate(movies):
        movies[i] = file[:-4]
    return(movies)


create_hdf5((112,112))
movies = get_movies()
for film in movies:
    store_film(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/',
        film, '.mp4')

'''
create_hdf5((212,212))
store_film('/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/','After_The_Rain','.mp4')
store_film('/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/','Attitude_Matters','.mp4')
movie1 = get_film('Attitude_Matters')
print(movie1.shape)
movie = get_film('After_The_Rain')
print(movie.shape)

print(get_dimensions2('Attitude_Matters'))
print(get_dimensions())
'''