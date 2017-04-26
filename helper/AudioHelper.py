"""
This assistant allows to compute different properties 
from audio such as duration and to extract the audio files from video.
"""


import subprocess
from helper import DatasetManager as Db
from scipy.io import wavfile


def extract_audio_from_video(video_path: str, audio_path: str, audio_name: str):
    command = 'ffmpeg -i ' + video_path + ' -ab 2 -ar 44100 -vn ' \
              + audio_path + '/' + audio_name + '.wav'
    subprocess.call(command, shell=True)


def extract_audios():
    videos_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/' \
                 'continuous-movies/'
    audio_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/audios'
    movies = Db.get_movies_names()
    for movie in movies:
        video_path = videos_path + movie + '.mp4'
        extract_audio_from_video(video_path, audio_path, movie)


def get_audio(movie: str):
    audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/audios/'
    fs, data = wavfile.read(audios_path + movie + '.wav')
    return data


def get_sampling_frequancy(movie: str):
    audios_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/audios/'
    fs = wavfile.read(audios_path + movie + '.wav')
    return fs[0]
