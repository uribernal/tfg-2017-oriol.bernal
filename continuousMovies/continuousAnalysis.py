"""
 The 1 second long segments used in [Baveye 2015-2] can be generated using the script: "ACCEDEsplitMoviesForContinuousAnalysis.py"
	  Please note that for this script, FFMPEG should be installed and in your working environment in order to be recognized by Python.
"""

import subprocess
import os
import re
import math
from datetime import timedelta

continuousAnnotationsFolder = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-annotations/continuous-annotations/'  # /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-annotations/continuous-annotations/
movieFolder = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/'  # /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/
outputFolder = '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/output/'


# Extract the duration of a movie
# /!\ Requires FFMPEG
def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout, stderr) = result.communicate()

    entries = stdout.decode('utf-8').split('\n')
    for line in entries:
        if (re.search('Duration', line)):
            # print(line)

            segments = line.split(',')
            duration = segments[0].replace('  Duration: ', '')

            return duration


# Extract the duration of a movie
# /!\ Requires FFMPEG
def cutMovie(moviePath, startSecond, endSecond, outputPath):
    # Example: ffmpeg -i input.mp4 -ss 00:00:30.0 -to 00:00:40.0 -y output.mp4
    commandLine = 'ffmpeg -loglevel error -i "' + moviePath + '" -ss ' + str(startSecond) + ' -to ' + str(
        endSecond) + ' -y "' + outputPath + '"'

    os.system(commandLine)


if __name__ == '__main__':

    movieNames = []

    for file in os.listdir(continuousAnnotationsFolder):
        if file.endswith(".txt"):

            movieName = ''

            if (re.search('_Arousal.txt', file)):
                movieName = file.replace('_Arousal.txt', '')
            elif (re.search('_Valence.txt', file)):
                movieName = file.replace('_Valence.txt', '')

            if (movieName not in movieNames):
                movieNames.append(movieName)

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    for movie in movieNames:

        moviePath = movieFolder + movie + '.mp4'

        # Check if movie file exists
        if (os.path.isfile(moviePath)):
            # print(moviePath)

            # Split the movie in small segments every second (each one second in duration)
            movieDuration = getLength(moviePath)

            movieDurations = movieDuration.split(':')
            movieTimeDelta = timedelta(hours=int(movieDurations[0]), minutes=int(movieDurations[1]),
                                       seconds=int(math.floor(float(movieDurations[2]))))

            movieTotalSeconds = movieTimeDelta.total_seconds()

            print('Splitting ' + moviePath + ' in ' + str(int(movieTotalSeconds)) + ' files... Please Wait.')#### change movieTotalSeconds per movie total frames

            for idx in range(int(movieTotalSeconds)):

                charName = ''

                if (idx < 10):
                    charName = '0000'
                elif (idx < 100):
                    charName = '000'
                elif (idx < 1000):
                    charName = '00'
                elif (idx < 10000):
                    charName = '0'

                outputPath = outputFolder + movie + '_' + charName + str(idx) + '.mp4'
                cutMovie(moviePath, timedelta(seconds=idx), timedelta(seconds=idx + 1), outputPath)

        else:
            print('ERROR: file ' + moviePath + ' does not exist')