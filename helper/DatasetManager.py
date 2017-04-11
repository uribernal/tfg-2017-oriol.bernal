"""
This assistant ...
"""

import h5py
import numpy as np


def vector_transform(vector1, l2):

    l1 = vector1.shape[0]
    l3 = vector1.shape[1]
    v = np.zeros((l2, l3))
    if l1 > l2:
        num_units = int(l1 / l2)
        res = int(l1 % l2)
        num_jumps = num_units
        vector2 = vector1[0:l1-res, :]
        for i in range(l2):
            for k in range(l3):
                x = vector2[0:num_jumps, k]
                v[i, :] = np.sum(x) / num_units
    elif l1 == l2:
        v = vector1
    else:
        raise Exception('vector1 mus be larger!')
    return v


def get_labels(film_name: str):
    """ Returns the labels of an specific movie
    Args:
        film_name (string): name of the film to be retrieved
    """

    database_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/' \
                    'final/labels_1frame.h5'

    with h5py.File(database_path, 'r') as hdf:
        # Read movie
        ar = hdf.get('LABELS/AROUSAL/' + film_name)
        val = hdf.get('LABELS/VALENCE/' + film_name)
        arousal = np.array(ar)
        valence = np.array(val)
        return arousal, valence


def get_data_and_labels(movies, c3d_path):
    with h5py.File(c3d_path, 'r') as hdf:
        # Read DB
        data = np.array(hdf.get('features'))
    labels = np.array([])
    for movie in movies:
        valence, arousal = get_labels(movie)
        labels = np.append(labels, valence, axis=0)

    return data, labels


def save_data_and_labels(movies, c3d_path):
    data = np.array([])
    labels = np.array([])
    for cont, movie in enumerate(movies):
        print('Processing film: {} --------------> {}'.format(movie, cont))
        valence, arousal = get_labels(movie)
        labels = np.append(labels, valence, axis=0)
        with h5py.File(c3d_path, 'r') as hdf:
            # Read DB
            a = hdf.get('features')

            # Get array for the movie
            film = a.get(movie)
            print('Film shape: {}'.format(film.shape))
            print('Labels shape: {}'.format(valence.shape))

            # Reshape to adapt
            v = vector_transform(film, valence.shape[0])
            print('New film shape: {}'.format(v.shape))

            # Concatenate movies
            data = np.append(data, v)

    print('--------------------------')
    cte = int(data.shape[0]/4096)
    data = data.reshape(cte, 4096)
    # Create the HDF5 file
    hdf = h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5', 'w')
    # Create the structure of the DB
    hdf.close()
    with h5py.File(
            '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/C3D_features_myData_resized.h5',
            'r+') as hdf:
        # Store data
        hdf.create_dataset('features', data=data, compression='gzip', compression_opts=9)
    print('DATA Stored:')


def get_data(movies, c3d_path):
    data = np.array([])
    for movie in movies:
        with h5py.File(c3d_path, 'r') as hdf:
            # Read DB
            valence, arousal = get_labels(movie)
            film = np.array(hdf.get('features/'+movie))
            v = vector_transform(film, valence.shape[0])
            data = np.append(data, v)
    cte = int(data.shape[0] / 201)
    data = data.reshape(cte, 201)
    # Create the HDF5 file
    hdf = h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/'
        'temporal_localitzation_output_myData_resized.h5', 'w')
    # Create the structure of the DB
    hdf.close()
    with h5py.File(
            '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/final/'
            'temporal_localitzation_output_myData_resized.h5',
            'r+') as hdf:
        # Store data
        hdf.create_dataset('features', data=data, compression='gzip', compression_opts=9)
    print('DATA Stored:')
    return data


def get_film(film_name: str):
    """ Returns the RGB frames of a movie
    Args:
        film_name (string): name of the film to be retrieved
    """

    database_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/' \
                    'final/labels_1frame.h5'
    with h5py.File(database_path, 'r') as hdf:
        # Read movie
        movies = hdf.get('MOVIES')

        data = movies.get(film_name)
        movie = np.array(data)
        return movie


def get_dimensions():
    """ Returns the weght and width of the movies """

    database_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/' \
                    'final/labels_1frame.h5'
    with h5py.File(database_path, 'r') as hdf:
        # Read movie
        movies = hdf.get('MOVIES')
        v = list(movies.attrs.values())
        return int(v[0]), int(v[1])


def get_movies_names():
    mypath = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
             'LIRIS-ACCEDE-continuous-movies/continuous-movies/'
    from os import listdir
    from os.path import isfile, join
    movies = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i, file in enumerate(movies):
        movies[i] = file[:-4]
    return movies
