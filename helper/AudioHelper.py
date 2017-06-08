"""
This assistant allows to compute different properties 
from audio such as duration and to extract the audio files from video.
"""

import subprocess
from helper import DatasetManager as Dm
import numpy as np
from scipy.io import wavfile


def extract_audio_from_video(video_path: str, audio_path: str):
    command = 'ffmpeg -i ' + video_path + ' -ab 2 -ar 44100 -vn ' \
              + audio_path
    subprocess.call(command, shell=True)


def extract_audios(videos_path=None, audios_path=None, videos_extension=None, audios_extension=None):
    if videos_path is None:
        from helper.DatasetManager import videos_path
    if audios_path is None:
        from helper.DatasetManager import audios_path
    if videos_extension is None:
        from helper.DatasetManager import videos_extension
    if audios_extension is None:
        from helper.DatasetManager import audios_extension

    movies = Dm.get_movies_names()
    for movie in movies:
        input = videos_path + movie + videos_extension
        output = audios_path + movie + audios_extension
        extract_audio_from_video(input, output)


def get_audio(movie_path: str):
    _, data = wavfile.read(movie_path)

    return data


def get_sampling_frequency(movie: str):
    from helper.DatasetManager import audios_path, audios_extension

    fs = wavfile.read(audios_path + movie + audios_extension)
    return fs[0]


def get_audio_samples(video: str):
    from helper.DatasetManager import audios_path, audios_extension

    samp = get_audio(audios_path + video + audios_extension).shape[0]
    return samp


def get_resized_audio(index_video: int, win_frames: int, print_info=False):
    from helper import DatasetManager as Dm
    movies = Dm.get_movies_names()
    predictions_length = Dm.get_predictions_length(movies)
    audio_path = Dm.get_audio_path(movies[index_video])
    sampl_freq, audio_array = wavfile.read(audio_path)
    nb_samples = audio_array.shape[0]
    if print_info:
        print('{0} shape: {1}'.format(movies[index_video], audio_array.shape))
    nb_frames = nb_samples // win_frames

    audio_array = audio_array[:nb_frames * win_frames, :]
    if print_info:
        print('{0} shape: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = audio_array.reshape((nb_frames, win_frames, 2))
    if print_info:
        print('resized {0}: {1}\n'.format(movies[index_video], audio_array.shape))

    cte = predictions_length[index_video] * 5
    audio_array = audio_array[:cte, :, :]
    audio_array = audio_array.reshape(int(audio_array.shape[0] / 5), 5, win_frames, 2)
    if print_info:
        print('resized {0}: {1}\n'.format(movies[index_video], audio_array.shape))
    return audio_array


def get_resized_audio2(index_video: int, win_frames, print_info=False):
    movies = Dm.get_movies_names()
    audio_path = Dm.get_audio_path(movies[index_video])
    sampl_freq, audio_array = wavfile.read(audio_path)
    nb_samples = audio_array.shape[0]
    if print_info:
        print('{0} shape: {1}'.format(movies[index_video], audio_array.shape))
    nb_frames = nb_samples // win_frames
    audio_array = audio_array[:nb_frames * win_frames, :]
    if print_info:
        print('{0} shape: {1}'.format(movies[index_video], audio_array.shape))

    audio_array = audio_array.reshape((nb_frames, win_frames, 2))
    if print_info:
        print('resized {0}: {1}\n'.format(movies[index_video], audio_array.shape))

    return audio_array.transpose(2, 0, 1)


def get_resized_audios(videos: list, win_frames, print_info=False):
    res = np.array([])
    for cont, video in enumerate(videos):
        resized_audio = get_resized_audio(cont, win_frames=win_frames, print_info=print_info)
        res = np.append(res, resized_audio)
    return res.reshape(int(res.shape[0] / (5 * win_frames * 2)), 5, win_frames, 2)


def compute_STFT_and_MelBank(data, print_info=False):
    pre_emphasis = 0.97
    frame_size = 8.0/735  # time windowing
    frame_stride = 4.0/735  # time overlapping
    sample_rate = 44100  # fs
    nfilt = 64  # number of filters
    NFFT = 512  # points for the STFT

    input = np.array([])

    # Pre-emphasis
    for j in range(data.shape[1]):
        signal = data[0, j, :]
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        if print_info:
            print('emphasized_signal: {}'.format(emphasized_signal.shape))

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
        if print_info:
            print('Frames: {}'.format(frames.shape))


        # Window
        frames *= np.hamming(frame_length)
        if print_info:
            print('Frames- window: {}'.format(frames.shape))

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
        if print_info:
            print('filter_banks: {}'.format(filter_banks.shape))

        input = np.append(input, filter_banks)
        print_info=False
    input = input.reshape(int(input.shape[0] / (96 * 64)), 96, 64)
    if print_info:
        print('Input: {}'.format(input.shape))

    return input


def get_acoustic_data(videos: list, win_frames, print_info=False):

    movies = Dm.get_movies_names()
    predictions_length = Dm.get_predictions_length(movies)

    resized_audios = np.array([])
    for cont, video in enumerate(videos):
        resized_audio = get_resized_audio2(cont, win_frames=win_frames, print_info=False)
        computed_audio = compute_STFT_and_MelBank(resized_audio, print_info=False)
        cte = predictions_length[cont] * 5
        computed_audio = computed_audio[:cte, :, :]
        computed_audio = computed_audio.reshape(int(computed_audio.shape[0] / 5), 5, 96, 64)
        resized_audios = np.append(resized_audios, computed_audio)

    resized_audios = resized_audios.reshape(int(resized_audios.shape[0]/(96*64*5)), 5, 96, 64)
    if print_info:
        print('resized: {0}\n'.format(resized_audios.shape))

    return resized_audios
