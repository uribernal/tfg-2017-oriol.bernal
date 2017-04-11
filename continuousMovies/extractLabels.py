import os

from continuousMovies.featureExtraction import VideoGenerator as vg
from continuousMovies.fine_tuning.albertoswork import video_extraxting_fps2 as db

raw_labels_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
                  'LIRIS-ACCEDE-continuous-annotations/continuous-annotations/raw/'

continuous_movies_path = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/' \
                         'LIRIS-ACCEDE-continuous-movies/continuous-movies/'

extension = '.mp4'

movies = db.get_movies_names()
ids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']
id = ids[0]###############################
movie = movies[0]#########################
for file in os.listdir(raw_labels_path+'/'+id):
    if file.endswith(movie+"_Valence_"+id+".txt"):
        print('1')


frames = []
duration = []
fps = []
rate = []
for i, movie in enumerate(movies):
    frames.append(vg.get_num_frames(continuous_movies_path+movie+extension))
    duration.append(vg.get_duration(continuous_movies_path+movie+extension))
    fps.append(frames[i]/duration[i])
    rate.append(1/frames[i])

