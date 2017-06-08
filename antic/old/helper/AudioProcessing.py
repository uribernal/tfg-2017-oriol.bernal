import h5py
import helper.DatasetManager as Db
import numpy as np
from scipy.io import wavfile

import old.helper.AudioHelper as Ah


def get_audio_resized(movie: str):
    a = Db.get_audio(movie)
    fs = Ah.get_sampling_frequency(movie)
    len = a.shape[0]
    q = len % 44100
    a = a[0:len - q, :]
    len = a.shape[0]
    cte = int(len / fs)
    k = a.reshape(2, cte, fs)

    return k

hdf = h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/audio/audio_data_fft.h5', 'w')
# Create the structure of the DB
hdf.close()

hdf = h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/audio/audio_data_fft_all.h5', 'w')
# Create the structure of the DB
hdf.close()

# Get list with the names of the movies
movies = Db.get_movies_names()

audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/audios/'

pre_emphasis = 0.97
frame_size = 0.025  # time windowing
frame_stride = 0.01  # time overlapping
sample_rate = 44100  # fs
nfilt = 64  # number of filters
NFFT = 512  # points for the STFT

cte = 0
input = np.array([])
input1 = np.array([])
for i, movie in enumerate(movies):
    input2 = np.array([])

    print('------------------------->{}'.format(i))
    fs, data = wavfile.read(audios_path + movie + '.wav')
    data = get_audio_resized(movie)

    len = data.shape[1]*data.shape[2]
    duration = len / fs

    print('Movie: {}'.format(movie))
    print('Duration: {}'.format(duration))
    print('Fs: {}'.format(fs))
    print('Shape: {}'.format(data.shape))

    # Pre-emphasis
    for j in range(data.shape[1]):
        signal = data[0, j, :]
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        #print('emphasized_signal: {}'.format(emphasized_signal.shape))

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
        #print('Frames: {}'.format(frames.shape))


        # Window
        frames *= np.hamming(frame_length)
        #print('Frames- window: {}'.format(frames.shape))

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
        #print('filter_banks: {}'.format(filter_banks.shape))

        input = np.append(input, filter_banks)
    #cte += j + 1
    input = input.reshape(int(input.shape[0]/(98*64)), 98, 64)
    input1 = np.append(input1, input)
    print('Input: {}'.format(input.shape))
    with h5py.File(
            '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/audio/audio_data_fft.h5',
            'r+') as hdf:
        # Store data
        hdf.create_dataset('features/' + movie, data=input, compression='gzip', compression_opts=9)
    print(movie + ' Stored: {}'.format(input.shape))
    input2 = np.append(input2, input1)
    input = np.array([])

print('\nInput2: {}'.format(input2.shape))
input2 = input2.reshape(int(input2.shape[0] / (98 * 64)), 98, 64)
with h5py.File(
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/audio/audio_data_fft_all.h5',
        'r+') as hdf:
    # Store data
    hdf.create_dataset('features', data=input2, compression='gzip', compression_opts=9)
print('All Stored: {}'.format(input2.shape))
'''
    a = Db.get_audio(movie)
    fs = Ah.get_sampling_frequancy(movie)
    len = a.shape[0]
    duration = len / fs

    # Show Movie info
    print('------------------------->{}'.format(i))
    print('Movie: {}'.format(movie))
    print('Duration: {}'.format(duration))
    print('Fs: {}'.format(fs))

    len = a.shape[0]
    q = len % 44100
    a = a[0:len-q,:]
    print('Audio shape: {}'.format(a.shape))

    valence, arousal = Db.get_labels(movie)
    labels = valence.shape[0] - 1

    print('Labels: {}'.format(labels))

    len = a.shape[0]
    cte = int(len/fs)

    k = a.reshape(2, fs, cte)
    print('K shape: {}\n'.format(k.shape))

    print(k[0, :, 0].shape)
    fft1 = fft(k[0, :, 0], n=4410)
    print(fft1)
    plt.plot(abs(fft1),'r')
    plt.show()

    if i == 0:
        break

'''