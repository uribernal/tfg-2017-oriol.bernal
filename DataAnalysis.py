import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


movies = ['After_The_Rain', 'Attitude_Matters', 'Barely_legal_stories', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Cloudland', 'Damaged_Kung_Fu', 'Decay', 'Elephant_s_Dream', 'First_Bite', 'Full_Service', 'Islands', 'Lesson_Learned', 'Norm', 'Nuclear_Family', 'On_time', 'Origami', 'Parafundit', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', 'Superhero', 'Tears_of_Steel', 'The_room_of_franz_kafka', 'The_secret_number', 'To_Claire_From_Sonny', 'Wanted', 'You_Again']
db_path = 'C:/Users/Uri/Desktop/labels.h5'


def plot_fear_flow(movie, db_path=None, plot=True, save_path=None):
    if db_path is None:
        from helper.DatasetManager import data_path as db_path

    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get('/dev/ground_truth_data/' + movie))
    # Show Fear for each film
    fig = plt.figure(0)
    fig.suptitle(movie, fontsize=14, fontweight='bold')
    x = np.arange(labels.shape[0])*5
    plt.plot(x, labels[:, 2], label='valence')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path+movie)
        plt.close()


def plot_valence_arousal_flow(movie, db_path=None, plot=True, save_path=None):
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
    plt.legend(loc='upper right')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path+movie)
        plt.close()


def plot_fps(videos, videos_path=None, videos_extension= None, plot=True, save_path=None):
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


def plot_durations(videos, videos_path=None, videos_extension= None, plot=True, save_path=None):
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
    fig.suptitle('FPS of the DB', fontsize=14, fontweight='bold')
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

db_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'
a = '/home/uribernal/Desktop/output/'
plot_ground_truth_histograms(movies, db_path=None, plot=False, save_path=a)

