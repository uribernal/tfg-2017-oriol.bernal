import numpy as np
import matplotlib.pyplot as plt
import subprocess #for extracting audio

#ANNOTATIONS
path_rankings = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt' #Rankings
path_sets = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt' #Sets

#FEATURES
path_featuresArousal = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-features/features/ACCEDEfeaturesArousal_TAC2015.txt' #Features for Arousal
path_featuresValence = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-features/features/ACCEDEfeaturesValence_TAC2015.txt' #Features for Valence

def getFeatures():
    np_featureValuesArousal = np.array([])  # 9800x23 (num_of_clips-features)
    np_featureValuesValence = np.array([])  # 9800x17 (num_of_clips-features)
    with open(path_featuresArousal) as f:
        featuresArousal = f.readline().strip().split()
        for line in f:
            featureValuesArousal = line.strip().split()
            a = np.array(featureValuesArousal[2:], dtype=float)  # dim = 23 (features)
            np_featureValuesArousal = np.append(np_featureValuesArousal, a, axis=0)
    with open(path_featuresValence) as f:
        featuresValence = f.readline().strip().split()
        for line in f:
            featureValuesValence = line.strip().split()
            b = np.array(featureValuesValence[2:], dtype=float)  # dim = 17 (features)
            np_featureValuesValence = np.append(np_featureValuesValence, b, axis=0)
    return (featuresArousal, np.reshape(np_featureValuesArousal, (9800,23)),featuresValence, np.reshape(np_featureValuesValence, (9800,17))) #np_featureValuesArousal[0] torna 23 features del primer video


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

def extractAudio():
    audio_path = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-data/audio'
    video_path = '/home/uribernal/Desktop/MediaEval2016/devset/clips/LIRIS-ACCEDE-data/data'
    ids, names, labels = getLabels()
    for i in names:
        command = 'ffmpeg -i '+ video_path + '/' + i + ' -ab 2 -ar 44100 -vn ' + audio_path + '/' + i[:-3]+'wav'
        print(i)
        subprocess(command, shell=True)




