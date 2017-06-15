import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


movies = ['After_The_Rain', 'Attitude_Matters', 'Barely_legal_stories', 'Between_Viewings', 'Big_Buck_Bunny', 'Chatter', 'Cloudland', 'Damaged_Kung_Fu', 'Decay', 'Elephant_s_Dream', 'First_Bite', 'Full_Service', 'Islands', 'Lesson_Learned', 'Norm', 'Nuclear_Family', 'On_time', 'Origami', 'Parafundit', 'Payload', 'Riding_The_Rails', 'Sintel', 'Spaceman', 'Superhero', 'Tears_of_Steel', 'The_room_of_franz_kafka', 'The_secret_number', 'To_Claire_From_Sonny', 'Wanted', 'You_Again']
db_path = 'C:/Users/Uri/Desktop/labels.h5'


def plot_fear_flow(movie, plot=True, save_path=None):
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get(movie))
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


def plot_valence_arousal_flow(movie, plot=True, save_path=None):
    with h5py.File(db_path, 'r') as hdf:
        labels = np.array(hdf.get(movie))
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
    plt.plot(x, fps, label='valence')
    plt.legend(loc='upper right')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + video)
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
    plt.plot(x, duration, label='valence')
    plt.legend(loc='upper right')
    if plot:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path + video)
        plt.close()