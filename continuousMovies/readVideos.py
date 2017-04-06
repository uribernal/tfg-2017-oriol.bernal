#http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
#https://help.ubuntu.com/community/OpenCV
FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
import numpy as np
import subprocess as sp
import json
import requests

#ANNOTATIONS
path_rankings = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt' #Rankings
path_sets = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt' #Sets

#FEATURES
path_featuresArousal = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-features/features/ACCEDEfeaturesArousal_TAC2015.txt' #Features for Arousal
path_featuresValence = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-features/features/ACCEDEfeaturesValence_TAC2015.txt' #Features for Valence

def getLabels():
    ids = []
    names = []
    valenceRank = []
    arousalRank = []
    valenceValue = []
    arousalValue = []
    valenceVariance = []
    arousalVariance = []
    sets = []
    with open(path_rankings) as f:
        for line in f:
            l = line.strip().split()
            ids.append(l[0])
            names.append(l[1])
            valenceRank.append(l[2])
            arousalRank.append(l[3])
            valenceValue.append(l[4])
            arousalValue.append(l[5])
            valenceVariance.append(l[6])
            arousalVariance.append(l[7])
        ids.pop(0)
        names.pop(0)
        valenceRank.pop(0)
        arousalRank.pop(0)
        valenceValue.pop(0)
        arousalValue.pop(0)
        valenceVariance.pop(0)
        arousalVariance.pop(0)
        labels = np.array([valenceRank,arousalRank,valenceValue,arousalValue,valenceVariance,arousalVariance])
    return (ids, names, labels.T)

def getVideo(film: str, nb_frames: int, height: int, weight: int):
    video = np.array([])
    command = [FFMPEG_BIN,
               '-i',
               film,
               # 624x352
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
    for i in range(0, nb_frames):
        print(i)
        raw_image = pipe.stdout.read(height * weight * 3)
        # transform the byte read into a numpy array
        image = np.fromstring(raw_image, dtype='uint8')
        try:
            image = image.reshape((height, weight, 3))
        except:
            image = np.zeros((height, weight, 3))
        video = np.append(video, image)
        # throw away the data in the pipe's buffer.
        pipe.stdout.flush()
    video = video.reshape((height, weight, 3, nb_frames))
    return video

def getVideos(films: list, nb_frames: int, height: int, weight: int, num_films=None):
    if num_films is None:
        num_films = len(films)
    videos = np.array([])
    for i, film in enumerate(films):
        print (i, film)
        a = getVideo(film, nb_frames, height, weight)
        videos = np.append(videos, a)
        if (i>=num_films-1):
            break
    videos = videos.reshape((nb_frames, num_films, height, weight, 3))
    return videos

def getLongMovies():
    films = [
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/After_The_Rain.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Attitude_Matters.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Barely_legal_stories.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Between_Viewings.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Big_Buck_Bunny.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Chatter.mp4',
        '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Cloudland.mp4']
    nb_frames = 50
    height = 352
    weight = 624
    videos = getVideos(films, nb_frames, height, weight)
    print(videos.shape)
    print('FIN')
    return videos

def getClips(num_clips=None):
    ids, names, labels = getLabels()
    src = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-data/data/'
    films = []
    for name in names:
        films.append(src+name)
    print(len(films))
    nb_frames = 1
    height = 224
    weight = 224
    videos = getVideos(films, nb_frames, height, weight, num_films=num_clips)
    print(videos.shape)
    print('FIN')
    return videos

def getInfoVideo(fn):
    import cv2
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print("could not open :",fn)
        return
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, length, fps





#a = getVideoInfo('/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-data/data/ACCEDE00007.mp4')
#print(a)
#a = getClips()
#print(a.shape)


#GET DIMENSIONS OF THE VIDEO
    # ffprobe -v error -show_entries stream=width,height \>   -of default=noprint_wrappers=1 /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Spaceman.mp4
#GET NUMBER OF FRAMES
    # ffprobe -select_streams v -show_streams /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Spaceman.mp4