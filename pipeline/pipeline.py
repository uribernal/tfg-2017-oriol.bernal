import numpy as np


def get_movies_names(path: str):
    from os import listdir
    from os.path import isfile, join

    videos = [f for f in listdir(path) if isfile(join(path, f))]
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


def get_videos_info(videos: list, ext: str):
    from helper import VideoHelper as Vh
    rgb_frames = []
    video_fps = []
    video_duration = []
    num_video_clips = []
    for cont, movie in enumerate(videos):
        rgb_frames.append(Vh.get_num_frames(
            videos_path + movie + ext))  # The number of frames per video (depend on its length and fps)
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
        audio_samples.append(Ah.get_audio(audios_path + movie + audios_extension)
                             .shape[0])  # The number of samples per audio (depend on its length and fs)
        audio_frames.append(float(audio_samples[cont] / win_frames))

    return sampling_freq, audio_samples, audio_frames


def read_video(input_video: str, resize: tuple=(112, 112)):
    from helper import VideoHelper as Vh
    vid = Vh.video_to_array(input_video, resize=resize)

    return vid


def get_visual_data(videos: list):
    data = np.array([])
    for cont, video in enumerate(videos):
        input_video = videos_path + video + videos_extension
        video_array = read_video(input_video, input_size)
        print('{0}: {1}'.format(video, video_array.shape))
        nb_frames = frames[cont]
        nb_clips = nb_frames // num_frames
        video_array = video_array.transpose(1, 0, 2, 3)
        video_array = video_array[:nb_clips * num_frames, :, :, :]
        video_array = video_array.reshape((nb_clips, num_frames, 3, 112, 112))
        video_array = video_array.transpose(0, 2, 1, 3, 4)
        print('resized {0}: {1}'.format(video, video_array.shape))
        data = np.append(data, video_array)

    return data.reshape(int(data.shape[0]/(3*16*112*112)), 3, 16, 112, 112)


def read_audio(input_video: str):
    from helper import AudioHelper as Ah
    aud = Ah.get_audio(input_video)

    return aud


def get_acoustic_data(videos: list):
    data = np.array([])
    for cont, video in enumerate(videos):
        input_video = audios_path + video + audios_extension
        audio_array = read_audio(input_video)
        print('{0}: {1}'.format(video, audio_array.shape))
        nb_samples = num_audio_samples[cont]
        nb_frames = nb_samples // win_frames
        # audio_array = audio_array.transpose(1, 0)
        audio_array = audio_array[:nb_frames * win_frames, :]
        audio_array = audio_array.reshape((nb_frames, win_frames, 2))
        # audio_array = audio_array.transpose(2, 0, 1)
        print('resized {0}: {1}'.format(video, audio_array.shape))
        data = np.append(data, audio_array)

    data = data.reshape(int(data.shape[0] / (2 * win_frames)), win_frames, 2)
    return data.transpose(0, 2, 1)


# PATHS
annotations_path = '/home/uribernal/Desktop/MediaEval2017/annotations/'

videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'LIRIS-ACCEDE-continuous-movies/continuous-movies/'

audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
              'audios/'

visual_features_path = ''
acoustic_features_path = ''
videos_extension = '.mp4'
audios_extension = '.wav'

# PRE-WORK
# Extract audios
# Change fps in videos

# CONSTANTS
time_prediction_constant = 10
time_shift_prediction_constant = 5
input_size = (112, 112)
num_frames = 16
num_visual_feat = 4096  # Features after C3D convolutions (visual feature extractor model)
num_acoustic_feat = 4096  # Features after CD convolutions (acoustic feature extractor model)
# fs = 44100  # Sampling frequency for audio files
win_frames = int(960 * 44100 * 1e-3)  # Length of window frames (no overlapping)
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
labels = get_ground_truth_data(movies)  # The ground truth data for each film
print('labels: ({0}, {1})\n'.format(labels.shape[0], labels.shape[1]))

print('\033[94m' + 'VISUAL INFORMATION:' + '\033[0m')
frames, fps, duration, num_clips = get_videos_info(movies, videos_extension)
print('frames ({0}): {1}'.format(len(frames), frames))
print('fps ({0}): {1}'.format(len(fps), fps))
print('duration ({0}): {1}'.format(len(duration), duration))
print('num_clips ({0}): {1}\n'.format(len(num_clips), num_clips))

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

print('\033[94m' + 'RGB DATA:' + '\033[0m')
visual_data = get_visual_data(movies[:0])  # Matrix with the RGB info from videos
print('visual_data ({0})\n'.format(visual_data.shape))

print('\033[94m' + 'AUDIO DATA:' + '\033[0m')
acoustic_data = get_acoustic_data(movies[:0])  # Matrix with the audio samples from videos  # dim = (num_films, 2, audio_length)
print('acoustic_data {0}'.format(acoustic_data.shape))

# PROCESSING
# process audio

'''
visual_features = np.array([])  # Matrix with the features from videos  # dim = (num_films, num_clips, num_frames, num_visual_feat)
acoustic_features = np.array([])  # Matrix with the audio features from videos  # dim = (num_films, num_audio_frames, num_stft, num_filter_banks)
'''
