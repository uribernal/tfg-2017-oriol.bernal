"""
This script allows to extract all the audios from the movies_path
or it extracts a single audio from movie_path into the audio_path
using the function extract_audio
"""

from helper import AudioHelper as Ah


def extract_audio(movie_path, audio_path, auido_name):
    Ah.extract_audio_from_video(movie_path, audio_path, auido_name)


if __name__ == '__main__':
    Ah.extract_audios()