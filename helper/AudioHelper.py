"""
This assistant allows to compute different properties 
from audio such as duration and to extract the audio files from video.
"""


import subprocess
from helper import DatasetManager as Db


def extract_audio_from_video(video_path, audio_path, audio_name):
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
