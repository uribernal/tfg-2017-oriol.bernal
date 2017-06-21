"""
This assistant allows to compute different properties 
from audio such as duration and to extract the audio files from video.
"""

import subprocess
from helper import DatasetManager as Dm
import numpy as np
from scipy.io import wavfile


def extract_audio_from_video(video_path: str, audio_path: str):
    """ Saves the audio files in audio_path extracted from the
    video located in video_path """

    command = 'ffmpeg -i ' + video_path + ' -ab 2 -ar 44100 -vn ' \
              + audio_path
    subprocess.call(command, shell=True)


def extract_audios(movies=None, videos_path=None, audios_path=None, videos_extension=None, audios_extension=None):
    """ Extract all the audios from the videos in the folder videos_path.
    movies is a list with the name of all the videos """

    if videos_path is None:
        from helper.DatasetManager import videos_path
    if audios_path is None:
        from helper.DatasetManager import audios_path
    if videos_extension is None:
        from helper.DatasetManager import videos_extension
    if audios_extension is None:
        from helper.DatasetManager import audios_extension
    if movies is None:
        movies = Dm.get_movies_names()

    for movie in movies:
        input = videos_path + movie + videos_extension
        output = audios_path + movie + audios_extension
        extract_audio_from_video(input, output)


def get_audio(audio_path: str):
    """ Returns the audio samples """

    _, data = wavfile.read(audio_path)

    return data


def get_sampling_frequency(audio: str):
    """ Returns the sampling frequency of the audio """

    from helper.DatasetManager import audios_path, audios_extension

    fs = wavfile.read(audios_path + audio + audios_extension)
    return fs[0]


def get_audio_samples(video: str):
    """ Returns the number of samples of the audio """

    from helper.DatasetManager import audios_path, audios_extension

    num_samples = get_audio(audios_path + video + audios_extension).shape[0]
    return num_samples


def get_resized_audio(audio_path: str, win_frames, print_info=False):
    """ Returns the audio in a matrix form (2, num_windows, samples_per_window).
        The audio is windowed and  2 is because of the Left/Right channels """

    sampl_freq, audio_array = wavfile.read(audio_path)
    nb_samples = audio_array.shape[0]
    if print_info:
        print('Audio shape: {0}'.format(audio_array.shape))
    nb_frames = nb_samples // win_frames
    audio_array = audio_array[:nb_frames * win_frames, :]
    if print_info:
        print('Windowed audio shape: {0}'.format(audio_array.shape))

    audio_array = audio_array.reshape((nb_frames, win_frames, 2))
    if print_info:
        print('Resized audio: {0}\n'.format(audio_array.shape))

    return audio_array.transpose(2, 0, 1)


def compute_STFT_and_MelBank(data, NFFT, nfilt, print_info=False):
    """ It computes the STFT (Fast Fourier Transform applied to windowed signals)
     and later it computes power banks using Mel filters """

    pre_emphasis = 0.97
    frame_size = 8.0/735  # time windowing
    frame_stride = 4.0/735  # time overlapping
    sample_rate = 44100  # fs
    #  nfilt = 64  # number of filters
    #  NFFT = 512  # points for the STFT

    computed_signal = np.array([])

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

        computed_signal = np.append(computed_signal, filter_banks)
        print_info=False
    computed_signal = computed_signal.reshape(int(computed_signal.shape[0] / (96 * 64)), 96, 64)
    if print_info:
        print('Computed signal: {}'.format(computed_signal.shape))

    return computed_signal

