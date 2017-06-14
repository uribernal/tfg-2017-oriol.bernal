"""
This assistant ...
"""
import numpy as np
from helper import AudioHelper as Ah
from keras import backend as K
K.set_image_dim_ordering('th')


# PATHS
annotations_path = '/home/uribernal/Desktop/MediaEval2017/annotations/'
videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-movies/continuous-movies/'
resized_videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
                      'LIRIS-ACCEDE-continuous-movies/30fps/'

audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'audios/'

data_path = '/home/uribernal/Desktop/MediaEval2017/data/data/data/emotional_impact.h5'

features_path = '/home/uribernal/Desktop/data/emotional_impact.h5'

visual_features_path = ''
acoustic_features_path = ''

videos_extension = '.mp4'
audios_extension = '.wav'

results_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/results.txt'

def get_video_path(video:str):
    return  videos_path + video + videos_extension


def get_audio_path(video:str):
    return  audios_path + video + audios_extension


def get_movies_names():
    from os import listdir
    from os.path import isfile, join

    videos = [f for f in listdir(videos_path) if isfile(join(videos_path, f))]
    for cont, file in enumerate(videos):
        videos[cont] = file[:-4]

    return videos


def get_movie_labels(path: str, movie: str):
    extensions = ['-MEDIAEVAL2017-valence_arousal.txt', '-MEDIAEVAL2017-fear.txt']
    time_1 = []
    mean_arousal = []
    mean_valence = []
    time_2 = []
    fear = []
    for cte, ext in enumerate(extensions):
        with open(path + movie + ext) as f:
            for line in f:
                if cte == 0:
                    l = line.strip().split()
                    time_1.append(l[1])
                    mean_valence.append(l[2])
                    mean_arousal.append(l[3])
                elif cte == 1:
                    l = line.strip().split()
                    time_2.append(l[1])
                    fear.append(l[2])
                else:
                    raise Exception('Error extracting labels')
        if cte == 0:
            time_1.pop(0)
            mean_valence.pop(0)
            mean_arousal.pop(0)
        elif cte == 1:
            time_2.pop(0)
            fear.pop(0)
        else:
            raise Exception('Error extracting labels')

    lab = np.array((mean_valence, mean_arousal, fear), dtype=float)
    return lab


def get_predictions_length(videos: list, show_info: str=False):
    res = []
    for video in videos:
        l = get_movie_labels(annotations_path, video)  # The ground truth data for each film
        res.append(l.shape[1])

    return res


def get_ground_truth_data(videos: list, show_info: str=False):
    lab = np.array([])
    for video in videos:
        l = get_movie_labels(annotations_path, video)  # The ground truth data for each film
        e = l.transpose(1, 0)
        lab = np.append(lab, e)
        if show_info:
            print('{0}: {1}'.format(video, l.shape))
    if show_info:
        print('')
    return lab.reshape(lab.shape[0]//3, 3)


def get_videos_info(videos: list):
    from helper import VideoHelper as Vh
    rgb_frames = []
    video_fps = []
    video_duration = []
    for cont, movie in enumerate(videos):
        rgb_frames.append(Vh.get_num_frames(
            videos_path + movie + videos_extension))  # The number of frames per video (depend on its length and fps)
        video_fps.append(Vh.get_fps(videos_path + movie + videos_extension))  # The number of fps per video
        video_duration.append(Vh.get_duration(videos_path + movie + videos_extension))

    return rgb_frames, video_fps, video_duration


def get_audio_info(video: str):
    sampling_freq = Ah.get_sampling_frequency(video)
    audio_samples = Ah.get_audio_samples(video)
    return sampling_freq, audio_samples


def get_audios_info(videos: list):
    sampling_freq = []
    audio_samples = []
    for cont, movie in enumerate(videos):
        sampling_freq.append(Ah.get_sampling_frequency(movie))
        audio_samples.append(Ah.get_audio(audios_path + movie + audios_extension)
                             .shape[0])  # The number of samples per audio (depend on its length and fs)

    return sampling_freq, audio_samples


def compute_pcc(y_pred, y_true):
    m1 = np.mean(y_pred)
    m2 = np.mean(y_true)
    y_pred_norm = y_pred - m1
    y_true_norm = y_true - m2
    nom = np.sum(y_pred_norm*y_true_norm)
    den = np.sqrt(np.sum(y_pred_norm**2))*np.sqrt(np.sum(y_true_norm**2))

    return nom/den


def expand_labels(labels):
    a1 = np.ones(9)
    a2 = np.ones(10)
    new_array = a1 * labels[0]

    sequence = [a1, a2, a1, a1, a2, a1, a2, a1]
    j = 1
    for i in range(labels.shape[0] - 1):
        if j == 8:
            j = 0
        cte = sequence[j]
        a = (cte * labels[i] + cte * labels[i + 1]) / 2.0
        new_array = np.append(new_array, a)
        j += 1
    if j == 8:
        j = 0
    a = sequence[j] * labels[-1]
    new_array = np.append(new_array, a)
    return new_array


def compress_labels(labels):
    a1 = 9
    a2 = 10

    sequence = [a1, a2, a1, a1, a2, a1, a2, a1]

    cont = True
    cte = a1
    new_array = [labels[0]]
    i = cte
    j = 1
    while cont:
        rest = labels.shape[0] - i
        if j == 8:
            j = 0
        cte = sequence[j]
        a = 2*labels[i]-new_array[-1]
        i += cte
        if rest <= 29:
            cont = False
        new_array = np.append(new_array, a)
        j += 1
    a = labels[-1]
    new_array = np.append(new_array, a)
    return new_array
