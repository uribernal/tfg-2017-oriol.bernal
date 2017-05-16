import numpy as np
import h5py

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


def get_predictions_length(videos: list, show_info: str=False):
    res = []
    for video in videos:
        l = get_movie_labels(annotations_path, video)  # The ground truth data for each film
        e = l.transpose(1, 0)
        res.append(e.shape[0])

    return res


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
        sampling_freq.append(Ah.get_sampling_frequency(movie))
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


def process_audio(data):
    pre_emphasis = 0.97
    frame_size = 0.025  # time windowing
    frame_stride = 0.01  # time overlapping
    sample_rate = 44100  # fs
    nfilt = 64  # number of filters
    NFFT = 512  # points for the STFT
    input = np.array([])
    input3 = np.zeros([])

    # Pre-emphasis
    #for s in range(data.shape[0]):
    for j in range(data.shape[1]):
        signal = ((data[0, j, :] + data[1, j, :]) / 2)
        # signal = data[s, j, :]
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        # print('emphasized_signal: {}'.format(emphasized_signal.shape))

        # Framing
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = emphasized_signal.shape[0]
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(
            float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal,
                               z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        # print('Frames: {}'.format(frames.shape))


        # Window
        frames *= np.hamming(frame_length)
        # print('Frames- window: {}'.format(frames.shape))

        # Fourier-Transform and Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        # Filter Banks
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        # print('filter_banks: {}'.format(filter_banks.shape))

        input = np.append(input, filter_banks)
    input = input.reshape(int(input.shape[0]/(98*64)), 98, 64)
    #print('Input: {}'.format(input.shape))
    return input


def get_audio_resized(video: str):
    from helper import AudioHelper as Ah

    a = Ah.get_audio(audios_path + video + audios_extension)
    fs = Ah.get_sampling_frequency(video)
    len = a.shape[0]
    q = len % 44100
    a = a[0:len - q, :]
    len = a.shape[0]
    cte = int(len / fs)
    k = a.reshape(2, cte, fs)

    return k


def get_audio_resized2(a, fs):
    len = a.shape[0]
    q = len % 44100
    a = a[0:len - q, :]
    len = a.shape[0]
    cte = int(len / fs)
    k = a.reshape(2, cte, fs)
    return k


def acoustic_data_processed(videos: list):
    from scipy.io import wavfile

    res = np.array([])
    for cont, video in enumerate(videos):
        data = get_audio_resized(video)
        sampl_freq, _ = wavfile.read(audios_path + video + '.wav')
        len = data.shape[1] * data.shape[2]
        duration = len / sampl_freq
        print('{0}: {1}'.format(video, data.shape))
        print('Duration: {}'.format(duration))
        print('Fs: {}'.format(sampl_freq))
        print('Shape: {}'.format(data.shape))

        processed_audio = process_audio(data)
        print('resized {0}: {1}\n'.format(video, processed_audio.shape))
        res = np.append(res, processed_audio)
    cte = int(res.shape[0] / (98 * 64))
    #cte2 = int(cte / 5)
    #aaa = cte % 5
    #len = cte2 - aaa
    #res = res[:len]

    #res = res.reshape(int(cte/cte2), cte2, 98, 64)
    res = res.reshape(cte, 98, 64)
    return res


def get_resized_video(index_video, video_path, input_size):
    video_array = read_video(video_path, input_size)
    print('{0}: {1}'.format(movies[index_video], video_array.shape))
    lab = predictions_length[index_video]
    resized_number_of_frames = lab * 5 * int(np.round(fps[index_video]))
    video_array = video_array[:, :resized_number_of_frames, :, :]
    video_array = video_array.transpose(1, 0, 2, 3)
    video_array = video_array.reshape((lab, int(video_array.shape[0]/lab), 3, input_size[0], input_size[1]))
    print('resized {0}: {1}'.format(movies[index_video], video_array.shape))
    return video_array


def get_resized_audio(index_video, audio_path):
    from scipy.io import wavfile

    sampl_freq, audio_array = wavfile.read(audio_path)
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))
    lab = predictions_length[index_video]
    resized_number_of_samples = lab * 5 * int(np.round(sampl_freq))
    audio_array = audio_array[:resized_number_of_samples, :]
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = get_audio_resized2(audio_array, sampl_freq)
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = process_audio(audio_array)
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = audio_array.transpose(1, 0, 2)
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = audio_array.reshape((lab, int(audio_array.shape[0] / lab), 2, 98, 64))


    len = audio_array.shape[1] * audio_array.shape[2]
    duration = len / sampl_freq
    print('{0}: {1}'.format(movies[index_video], audio_array.shape))

    print('Shape: {}'.format(audio_array.shape))

    processed_audio = process_audio(audio_array)
    print('resized {0}: {1}\n'.format(movies[index_video], processed_audio.shape))
    return processed_audio


def get_mixed_data(videos: list):
    res = np.array([])
    for cont, video in enumerate(videos):
        input_video = videos_path + video + videos_extension
        input_audio = audios_path + video + audios_extension

        #visual = get_resized_video(cont, input_video, (98, 64))
        acoustic = get_resized_audio(cont, input_audio)
    return res


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
predictions_length = get_predictions_length(movies)  # The number of predictions per video (depend on its length)
print('predictions_length ({0}): {1}\n'.format(len(predictions_length), predictions_length))
'''
print('\033[94m' + 'RGB DATA:' + '\033[0m')
visual_data = get_visual_data(movies[:0])  # Matrix with the RGB info from videos
print('visual_data ({0})\n'.format(visual_data.shape))

print('\033[94m' + 'AUDIO DATA:' + '\033[0m')
acoustic_data = get_acoustic_data(movies[:0])  # Matrix with the audio samples from videos
print('acoustic_data {0}\n'.format(acoustic_data.shape))
acoustic_data_processed = acoustic_data_processed(movies[:0])  # Matrix with the STFT-mel audio samples from videos
print('acoustic_data_processed {0}\n'.format(acoustic_data_processed.shape))

print('\033[94m' + 'MIXED DATA:' + '\033[0m')
data = get_mixed_data(movies[:2])


# PROCESSING
# process audio

'''
visual_features = np.array([])  # Matrix with the features from videos  # dim = (num_films, num_clips, num_frames, num_visual_feat)
acoustic_features = np.array([])  # Matrix with the audio features from videos  # dim = (num_films, num_audio_frames, num_stft, num_filter_banks)
'''

# STORE DATA

file = '/home/uribernal/Desktop/MediaEval2017/data/emotional_impact.h5'
store = False
if store:
    # Create the HDF5 file
    hdf = h5py.File(file, 'w')
    # Store data
    with h5py.File(file,'r+') as hdf:
        # Store data
        #hdf.create_dataset('movies', data=movies, dtype='<U23', compression='gzip', compression_opts=9)
        #hdf.create_dataset('labels_type', data=labels_type, compression='gzip', compression_opts=9)
        hdf.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        hdf.create_dataset('frames', data=frames, compression='gzip', compression_opts=9)
        hdf.create_dataset('fps', data=fps, compression='gzip', compression_opts=9)
        hdf.create_dataset('duration', data=duration, compression='gzip', compression_opts=9)
        hdf.create_dataset('fs', data=fs, compression='gzip', compression_opts=9)
        hdf.create_dataset('num_audio_samples', data=num_audio_samples, compression='gzip', compression_opts=9)
        hdf.create_dataset('predictions_length', data=predictions_length, compression='gzip', compression_opts=9)

        #hdf.create_dataset('visual_data', data=visual_data, compression='gzip', compression_opts=9)
        hdf.create_dataset('acoustic_data', data=acoustic_data, compression='gzip', compression_opts=9)
    print('DATA Stored:')


    # GET DATA
    with h5py.File(file, 'r') as hdf:
        # Read movie
        mov = hdf.get('acoustic_data')
        audio = np.array(mov)
    print(audio.shape)
'''