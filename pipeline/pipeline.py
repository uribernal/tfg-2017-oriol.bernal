import numpy as np


def get_movies_names(path: str):
    from os import listdir
    from os.path import isfile, join

    videos = [f for f in listdir(path) if isfile(join(path, f))]
    for cont, file in enumerate(videos):
        videos[cont] = file[:-4]

    return videos


def get_movie_labels(path: str, movie: str):
    extensions = ['_Arousal.txt', '_Valence.txt']
    time_arousal = []
    mean_arousal = []
    std_arousal = []
    time_valence = []
    mean_valence = []
    std_valence = []
    for cte, ext in enumerate(extensions):
        with open(path + movie + ext) as f:
            for line in f:
                if cte == 0:
                    l = line.strip().split()
                    time_arousal.append(l[0])
                    mean_arousal.append(l[1])
                    std_arousal.append(l[2])
                elif cte == 1:
                    l = line.strip().split()
                    time_valence.append(l[0])
                    mean_valence.append(l[1])
                    std_valence.append(l[2])
                else:
                    raise Exception('Error extracting labels')
        if cte == 0:
            time_arousal.pop(0)
            mean_arousal.pop(0)
            std_arousal.pop(0)
        elif cte == 1:
            time_valence.pop(0)
            mean_valence.pop(0)
            std_valence.pop(0)
        else:
            raise Exception('Error extracting labels')
    if time_arousal[-1] > time_valence[-1]:
        mean_arousal = mean_arousal[:len(mean_valence)]
    else:
        mean_valence = mean_valence[:len(mean_arousal)]
    lab = np.array((mean_arousal, mean_valence), dtype=float)
    return lab


def get_videos_info(videos: list, ext: str):
    from helper import VideoHelper as Vh
    rgb_frames = []
    video_fps = []
    video_duration = []
    num_video_clips = []
    for cont, movie in enumerate(videos):
        rgb_frames.append(Vh.get_num_frames(videos_path + movie + ext))  # The number of frames per video (depend on its length and fps)
        video_fps.append(Vh.get_fps(videos_path + movie + ext))  # The number of fps per video
        video_duration.append(Vh.get_duration(videos_path + movie + ext))
        num_video_clips.append(float(rgb_frames[cont] / num_frames))

    return rgb_frames, video_fps, video_duration, num_video_clips


def get_audios_info(videos: list):
    from helper import AudioHelper as Ah
    sampling_freq = []
    audio_samples = []
    audio_frames = []
    for cont, movie in enumerate(videos):
        sampling_freq.append(Ah.get_sampling_frequancy(movie))
        audio_samples.append(Ah.get_audio(movie).shape[0])  # The number of samples per audio (depend on its length and fs)
        audio_frames.append(float(audio_samples[cont] / win_frames))

    return sampling_freq, audio_samples, audio_frames


def read_video(input_video: str, input_size: tuple=(112, 112)):
    from helper import VideoHelper as Vh
    vid = Vh.video_to_array(input_video, resize=input_size)

    return vid


# PATHS
labels_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-annotations/continuous-annotations/'

videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-movies/continuous-movies/'

audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'audios/'

visual_features_path = ''
acoustic_features_path = ''
videos_extension = '.mp4'
audios_extension = '.wav'

# CONSTANTS
time_prediction_constant = 10
time_shift_prediction_constant = 5
input_size=(112, 112)
num_frames = 16
num_visual_feat = 4096  # Features after C3D convolutions (visual feature extractor model)
num_acoustic_feat = 4096  # Features after CD convolutions (acoustic feature extractor model)
# fs = 44100  # Sampling frequency for audio files
win_frames = 960 * 44100 * 1e-3  # Length of window frames (no overlapping)
num_stft = 98
num_filter_banks = 64

# THE TASK
print('\033[94m' + 'TASK INFORMATION:' + '\033[0m')
movies = get_movies_names(videos_path)  # Names of the videos from the DB
print('films ({0}): {1}'.format(len(movies), movies))
labels_type = ['valence', 'arousal', 'fear']  # Types of ground truth data per video
print('types of labels ({0}): {1}\n'.format(len(labels_type), labels_type))


# Calculate
print('\033[94m' + 'GROUND TRUTH INFORMATION:' + '\033[0m')
labels = get_movie_labels(labels_path, movies[1])  # The ground truth data for each film
print('labels ({0}, {1})\n'.format(len(labels), len(labels[0])))

print('\033[94m' + 'VISUAL INFORMATION:' + '\033[0m')
frames, fps, duration, num_clips = get_videos_info(movies, videos_extension)
print('frames ({0}): {1}'.format(len(frames), frames))
print('fps ({0}): {1}'.format(len(fps), fps))
print('duration ({0}): {1}'.format(len(duration), duration))
print('num_clips ({0}): {1}\n'.format(len(num_clips), num_clips))
'''
print('\033[94m' + 'ACOUSTIC INFORMATION:' + '\033[0m')
fs, num_audio_samples, num_audio_frames = get_audios_info(movies)
print('fs ({0}): {1}'.format(len(fs), fs))
print('num_audio_samples ({0}): {1}'.format(len(num_audio_samples), num_audio_samples))
print('num_audio_frames ({0}): {1}\n'.format(len(num_audio_frames), num_audio_frames))

print('\033[94m' + 'PREDICTION INFORMATION:' + '\033[0m')
predictions_length = []  # The number of predictions per video (depend on its length)
for i, _ in enumerate(fps):
    predictions_length.append(float(duration[i]/time_shift_prediction_constant))
print('predictions_length ({0}): {1}\n'.format(len(predictions_length), predictions_length))
'''

print('\033[94m' + 'RGB VIDEOS:' + '\033[0m')
input_video = videos_path + movies[0] + videos_extension
video_array = read_video(input_video, input_size)
print('video shape: {0}'.format(video_array.shape))
nb_frames = frames[0]
nb_clips = nb_frames // num_frames
video_array = video_array.transpose(1, 0, 2, 3)
video_array = video_array[:nb_clips * num_frames, :, :, :]
video_array = video_array.reshape((nb_clips, num_frames, 3, 112, 112))
video_array = video_array.transpose(0, 2, 1, 3, 4)
print('resized video shape: {0}'.format(video_array.shape))

'''
visual_data = np.array([])  # Matrix with the RGB info from videos  # dim = (num_films, 3, frames, video_height, video_width)
acoustic_data = np.array([])  # Matrix with the audio samples from videos  # dim = (num_films, 2, audio_length)

visual_features = np.array([])  # Matrix with the features from videos  # dim = (num_films, num_clips, num_frames, num_visual_feat)
acoustic_features = np.array([])  # Matrix with the audio features from videos  # dim = (num_films, num_audio_frames, num_stft, num_filter_banks)
'''
