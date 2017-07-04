"""
This assistant allows to compute some features of the database.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def plot_fear_flow(movie, db_path=None, plot=True, save_path=None):
    """ Shows the fear values of a film (0 or 1 values)
    Example: plots for all the videos in the developing set (30 videos):
        docs / images / fear_flow.png
    """

    if db_path is None:
        from helper.DatasetManager import data_path as db_path

    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('/dev/ground_truth_data/' + movie))
    # Show Fear for each film
    fig = plt.figure(0)
    fig.suptitle(movie, fontsize=14, fontweight='bold')
    x = np.arange(labels.shape[0])*5
    plt.plot(x, labels[:, 2], label='valence')
    plt.xlabel('time')
    plt.ylabel('classes')
    if save_path is not None:
        plt.savefig(save_path + movie + '_fear-flow.png')
        plt.close()

    if plot:
        plt.show()


def plot_valence_arousal_flow(movie, db_path=None, plot=True, save_path=None):
    """ Shows the valence and arousal continuous values of a film
    Example: plots for all the videos in the developing set (30 videos):
        docs / images / valence_and_arousal_flow.png
    """
    if db_path is None:
        from helper.DatasetManager import data_path as db_path

    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('/dev/ground_truth_data/' + movie))
    # Show Fear for each film
    fig = plt.figure(0)
    fig.suptitle(movie, fontsize=14, fontweight='bold')
    x = np.arange(labels.shape[0])*5
    plt.plot(x, labels[:, 0], label='valence')
    plt.plot(x, labels[:, 1], label='arousal')
    plt.xlabel('time')
    plt.ylabel('scores')
    plt.legend(loc='upper right')
    if save_path is not None:
        plt.savefig(save_path + movie + '_valence-arousal-flow.png')
        plt.close()

    if plot:
        plt.show()


def plot_fps(videos, videos_path=None, videos_extension=None, plot=True, save_path=None):
    """ Shows the frames per second of every video in the list: videos
    Example: plots for all the videos in the developing set (30 videos):
        docs / images / fps.png
        """
    from helper.VideoHelper import get_fps

    if videos_path is None:
        from helper.DatasetManager import videos_path
    if videos_extension is None:
        from helper.DatasetManager import videos_extension

    fps = []
    for video in videos:
        path = videos_path+video+videos_extension
        fps.append(get_fps(path))

    fig = plt.figure(0)
    fig.suptitle('FPS of the DB', fontsize=14, fontweight='bold')
    x = np.arange(len(fps))
    plt.bar(x, fps)
    plt.xlabel('movies')
    plt.ylabel('frames per second')
    plt.axis([-1, 30, 23, 31])
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + 'fps.png')
        plt.close()


def plot_durations(videos, videos_path=None, videos_extension=None, plot=True, save_path=None):
    """ Shows the duration of every video in the list: videos
    Example: plots for all the videos in the developing set (30 videos):
        docs / images / duration.png
        """
    from helper.VideoHelper import get_duration

    if videos_path is None:
        from helper.DatasetManager import videos_path
    if videos_extension is None:
        from helper.DatasetManager import videos_extension

    duration = []
    for video in videos:
        path = videos_path + video + videos_extension
        duration.append(get_duration(path))

    fig = plt.figure(0)
    fig.suptitle('Duration of the DB', fontsize=14, fontweight='bold')
    x = np.arange(len(duration))
    plt.bar(x, duration)
    plt.xlabel('movies')
    plt.ylabel('duration of each movie')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + 'duration.png')
        plt.close()


def plot_ground_truth_histograms(videos, db_path=None, plot=True, save_path=None):
    """ Shows the histogram values (valence, arousal and fear) of every video in the list: videos
    Examples: plots for all the videos in the developing set (30 videos):
    docs/images/arousal_histogram.png
    docs/images/valence_histogram.png
    docs/images/fear_histogram.png
    """

    if db_path is None:
        from helper.DatasetManager import data_path as db_path

    labels = np.array([])
    for video in videos:
        with h5py.File(db_path, 'r') as hdf:
            l = np.array(hdf.get('/dev/ground_truth_data/' + video))
            labels = np.append(labels, l)
    labels = labels.reshape(labels.shape[0]//3, 3)

    fig = plt.figure(0)
    fig.suptitle('Valence Histogram', fontsize=14, fontweight='bold')
    n, bins, patches = plt.hist(labels[:, 0], 100, normed=1, alpha=0.80, rwidth=0.7)
    y = mlab.normpdf(bins, np.mean(labels[:, 0]), np.std(labels[:, 0]))
    plt.plot(bins, y, 'r--', linewidth=1, )
    plt.xlabel('valence values')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + 'valence_histogram.png')
        plt.close()

    fig = plt.figure(1)
    fig.suptitle('Arousal Histogram', fontsize=14, fontweight='bold')
    n, bins, patches = plt.hist(labels[:, 1], 100, normed=1, alpha=0.80, rwidth=0.7)
    y = mlab.normpdf(bins, np.mean(labels[:, 1]), np.std(labels[:, 1]))
    plt.plot(bins, y, 'r--', linewidth=1, )
    plt.xlabel('arousal values')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + 'arousal_histogram.png')
        plt.close()

    fig = plt.figure(2)
    fig.suptitle('Fear Histogram', fontsize=14, fontweight='bold')
    plt.hist(labels[:, 2], 2, range=(-0.5, 1.5), rwidth=0.5)
    plt.xlabel('fear values')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + 'fear_histogram.png')
        plt.close()
