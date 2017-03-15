from xml.dom import minidom
import numpy as np
import pylab as pl
import re, fileinput
import os.path

# Input data
movieNames = ['After_The_Rain', 'Attitude_Matters', 'Barely_legal_stories', 'Between_Viewings', 'Big_Buck_Bunny',
              'Chatter', 'Cloudland', 'Damaged_Kung_Fu', 'Decay', 'Elephant_s_Dream', 'First_Bite', 'Full_Service',
              'Islands', 'Lesson_Learned', 'Norm', 'Nuclear_Family', 'On_time', 'Origami', 'Parafundit', 'Payload',
              'Riding_The_Rails', 'Sintel', 'Spaceman', 'Superhero', 'Tears_of_Steel', 'The_room_of_franz_kafka',
              'The_secret_number', 'To_Claire_From_Sonny', 'Wanted', 'You_Again']

ranksFile = '..\\annotations\\ACCEDEranking.txt'
databaseFilesXML = '..\\ACCEDEdescription.xml'
movieFilesXML = '..\\ACCEDEmovies.xml'
continuousAnnotationsFolder = '..\\continuous-annotations\\'
outputFolder = 'output\\curves\\'

if __name__ == '__main__':

    # Read databaseFilesXML
    databaseXMLdoc = minidom.parse(databaseFilesXML)
    idsList = databaseXMLdoc.getElementsByTagName('id')
    movieFileList = databaseXMLdoc.getElementsByTagName('movie')
    startFrameList = databaseXMLdoc.getElementsByTagName('start')
    endFrameList = databaseXMLdoc.getElementsByTagName('end')

    # Read movieFilesXML
    movieXMLdoc = minidom.parse(movieFilesXML)
    moviesList = movieXMLdoc.getElementsByTagName('movie')
    genresList = movieXMLdoc.getElementsByTagName('genre')
    excerptsList = movieXMLdoc.getElementsByTagName('excerpts')
    titleList = movieXMLdoc.getElementsByTagName('title')

    fpsMovie = [['After_The_Rain', 23.976],
                ['Attitude_Matters', 29.97],
                ['Barely_legal_stories', 23.976],
                ['Between_Viewings', 25],
                ['Big_Buck_Bunny', 24],
                ['Chatter', 24],
                ['Cloudland', 25],
                ['Damaged_Kung_Fu', 25],
                ['Decay', 23.976],
                ['Elephant_s_Dream', 24],
                ['First_Bite', 25],
                ['Full_Service', 29.97],
                ['Islands', 23.976],
                ['Lesson_Learned', 29.97],
                ['Norm', 25],
                ['Nuclear_Family', 23.976],
                ['On_time', 30],
                ['Origami', 24],
                ['Parafundit', 24],
                ['Payload', 25],
                ['Riding_The_Rails', 23.976],
                ['Sintel', 24],
                ['Spaceman', 23.976],
                ['Superhero', 29.97],
                ['Tears_of_Steel', 24],
                ['The_room_of_franz_kafka', 29.786],
                ['The_secret_number', 23.976],
                ['To_Claire_From_Sonny', 23.976],
                ['Wanted', 25],
                ['You_Again', 29.97]]

    # Statistics
    minContinuousArousal = ['', 1]
    maxContinuousArousal = ['', -1]
    meanContinuousArousal = 0
    meanDynamicContinuousArousal = 0

    minContinuousValence = ['', 1]
    maxContinuousValence = ['', -1]
    meanContinuousValence = 0
    meanDynamicContinuousValence = 0

    for movieName in movieNames:

        # Search for selected movie
        selectedNameList = []
        selectedNbList = []
        movieTitle = ''
        movieGenre = ''

        for idx, movieFile in enumerate(moviesList):
            found = re.search('^' + movieName + '.mp4',
                              movieFile.childNodes[0].nodeValue)  # parse for line with trim values
            if (found):
                selectedNameList.append(movieFile.childNodes[0].nodeValue)
                selectedNbList.append(excerptsList[idx].childNodes[0].nodeValue)
                movieTitle = titleList[idx].childNodes[0].nodeValue
                movieGenre = genresList[idx].childNodes[0].nodeValue

        # Select video clips
        selectedIds = []
        selectedMovies = []
        selectedStartFrames = []
        selectedEndFrames = []

        emptyNode = 0

        for idx, clipFile in enumerate(movieFileList):
            for selectedName in selectedNameList:
                if (selectedName == clipFile.childNodes[0].nodeValue):
                    selectedIds.append(idsList[idx].childNodes[0].nodeValue)
                    selectedMovies.append(selectedName)
                    # Check if frame values are available
                    if (len(startFrameList[idx].childNodes) == 1 and len(endFrameList[idx].childNodes) == 1):
                        selectedStartFrames.append(int(startFrameList[idx].childNodes[0].nodeValue))
                        selectedEndFrames.append(int(endFrameList[idx].childNodes[0].nodeValue))
                    else:
                        emptyNode = 1

        discreteAnnotationsValenceValues = []
        discreteAnnotationsArousalValues = []
        discreteAnnotationsStartSeconds = []
        discreteAnnotationsEndSeconds = []

        if (emptyNode):
            print('ERROR: frame values not available for "' + movieTitle + '"')

        else:
            print('Movie: ' + movieTitle)
            print('Files: ' + str(selectedNameList))
            print('Excerpts per file: ' + str(selectedNbList))

            # Modify frame values if the original movies is splitted into several files
            # /!\ NO LONGER NECESSARY WITH LAST UPDATE OF LIRIS-ACCEDE
            if (len(selectedNameList) > 1):
                selectedMovieNumber = []

                for clipMovie in selectedMovies:
                    cut = clipMovie.replace(movieName, '')
                    cut = cut.replace('.mp4', '')
                    entries = re.split('_', cut.strip())
                    selectedMovieNumber.append(int(entries[1]))

                sortedUniqueNumbers = sorted(list(set(selectedMovieNumber)))

                maxFrames = [0] * (int(max(sortedUniqueNumbers)) + 1)

                for number in sortedUniqueNumbers:
                    for idx, endFrame in enumerate(selectedEndFrames):
                        if (number == selectedMovieNumber[idx]):
                            if (maxFrames[number] < endFrame):
                                maxFrames[number] = endFrame

                for idx in range(len(maxFrames) - 1):
                    maxFrames[idx + 1] = maxFrames[idx + 1] + maxFrames[idx]

                minNumber = int(min(sortedUniqueNumbers))

                for idx, number in enumerate(selectedMovieNumber):
                    if (number > minNumber):
                        for idUnique, unique in enumerate(sortedUniqueNumbers):
                            if (unique == number):
                                selectedStartFrames[idx] = selectedStartFrames[idx] + maxFrames[idUnique]
                                selectedEndFrames[idx] = selectedEndFrames[idx] + maxFrames[idUnique]

            # Import rank values
            selectedArousalValues = []
            selectedValenceValues = []

            for line in fileinput.input(ranksFile):
                for ids in selectedIds:
                    readLine = re.search('^' + str(ids) + '\t', line)
                    if (readLine):
                        entries = re.split('\t', line.strip())
                        selectedValenceValues.append(float(entries[4]))
                        selectedArousalValues.append(float(entries[5]))

            if (len(selectedValenceValues) != len(selectedStartFrames)):
                print('ERROR')

            # Find fps
            fps = 25

            for couple in fpsMovie:
                if (couple[0] == movieName):
                    fps = float(couple[1])
                    print('Fps: ' + str(fps))

            # Convert frames in seconds and sort lists
            ind = np.argsort(selectedStartFrames)

            for idx in ind:
                discreteAnnotationsValenceValues.append((selectedValenceValues[idx] - 1) / 2 - 1)
                discreteAnnotationsArousalValues.append((selectedArousalValues[idx] - 1) / 2 - 1)
                discreteAnnotationsStartSeconds.append(int(selectedStartFrames[idx] / fps))
                discreteAnnotationsEndSeconds.append(int(selectedEndFrames[idx] / fps))

        continuousAnnotationsArousalValues = []
        continuousAnnotationsArousalStd = []
        continuousAnnotationsValenceValues = []
        continuousAnnotationsValenceStd = []
        continuousAnnotationsSeconds = []

        fileArousal = continuousAnnotationsFolder + movieName + '_Arousal.txt'
        fileValence = continuousAnnotationsFolder + movieName + '_Valence.txt'

        if not (os.path.isfile(fileArousal)):
            print('ERROR: file ' + fileArousal + ' does not exist')

        if not (os.path.isfile(fileValence)):
            print('ERROR: file ' + fileArousal + ' does not exist')

        if (os.path.isfile(fileValence) and os.path.isfile(fileArousal)):

            for line in fileinput.input(fileArousal):
                readLine = re.search('^[0-9]+\t', line)
                if (readLine):
                    entries = re.split('\t', line.strip())
                    continuousAnnotationsSeconds.append(int(entries[0]))
                    continuousAnnotationsArousalValues.append(float(entries[1]))
                    continuousAnnotationsArousalStd.append(float(entries[2]))

            for line in fileinput.input(fileValence):
                readLine = re.search('^[0-9]+\t', line)
                if (readLine):
                    entries = re.split('\t', line.strip())
                    continuousAnnotationsValenceValues.append(float(entries[1]))
                    continuousAnnotationsValenceStd.append(float(entries[2]))

            newLength = min(len(continuousAnnotationsValenceValues), len(continuousAnnotationsArousalValues))
            continuousAnnotationsSeconds = continuousAnnotationsSeconds[0:newLength]
            continuousAnnotationsArousalValues = continuousAnnotationsArousalValues[0:newLength]
            continuousAnnotationsArousalStd = continuousAnnotationsArousalStd[0:newLength]
            continuousAnnotationsValenceValues = continuousAnnotationsValenceValues[0:newLength]
            continuousAnnotationsValenceStd = continuousAnnotationsValenceStd[0:newLength]

            # Compute statistics

            if (min(continuousAnnotationsArousalValues) < minContinuousArousal[1]):
                minContinuousArousal[0] = movieName
                minContinuousArousal[1] = min(continuousAnnotationsArousalValues)

            if (min(continuousAnnotationsValenceValues) < minContinuousValence[1]):
                minContinuousValence[0] = movieName
                minContinuousValence[1] = min(continuousAnnotationsValenceValues)

            if (max(continuousAnnotationsArousalValues) > maxContinuousArousal[1]):
                maxContinuousArousal[0] = movieName
                maxContinuousArousal[1] = max(continuousAnnotationsArousalValues)

            if (max(continuousAnnotationsValenceValues) > maxContinuousValence[1]):
                maxContinuousValence[0] = movieName
                maxContinuousValence[1] = max(continuousAnnotationsValenceValues)

            meanContinuousArousal = meanContinuousArousal + np.mean(continuousAnnotationsArousalValues)
            meanContinuousValence = meanContinuousValence + np.mean(continuousAnnotationsValenceValues)

            meanDynamicContinuousArousal = meanDynamicContinuousArousal + max(continuousAnnotationsArousalValues) - min(
                continuousAnnotationsArousalValues)
            meanDynamicContinuousValence = meanDynamicContinuousValence + max(continuousAnnotationsValenceValues) - min(
                continuousAnnotationsValenceValues)

            # Plot the ranks for valence and arousal
            fig = pl.figure(figsize=(12, 12))

            # pl.subplot(211)
            pl.subplot2grid((4, 2), (0, 0), colspan=2)
            pl.title(movieTitle + '\n' + str(len(selectedIds)) + ' excerpts - ' + movieGenre)
            for idx in range(len(discreteAnnotationsStartSeconds)):
                exc = np.array([[discreteAnnotationsStartSeconds[idx], discreteAnnotationsValenceValues[idx]],
                                [discreteAnnotationsEndSeconds[idx], discreteAnnotationsValenceValues[idx]]])

                pl.plot(exc[:, 0], exc[:, 1], 'k', lw=4)

            pl.plot(np.asanyarray(continuousAnnotationsSeconds), np.asanyarray(continuousAnnotationsValenceValues), 'b',
                    lw=1, label='mean')
            pl.fill_between(np.asanyarray(continuousAnnotationsSeconds),
                            np.asanyarray(continuousAnnotationsValenceValues) - 1.96 * np.asanyarray(
                                continuousAnnotationsValenceStd),
                            np.asanyarray(continuousAnnotationsValenceValues) + 1.96 * np.asanyarray(
                                continuousAnnotationsValenceStd),
                            color='grey', alpha='0.3', label='95% confidence interval')
            pl.xlim([0, max(max(continuousAnnotationsSeconds), max(discreteAnnotationsEndSeconds))])
            pl.ylim([-1, 1])
            pl.ylabel('Valence')

            # pl.subplot(212)
            pl.subplot2grid((4, 2), (1, 0), colspan=2)
            for idx in range(len(discreteAnnotationsStartSeconds)):
                exc = np.array([[discreteAnnotationsStartSeconds[idx], discreteAnnotationsArousalValues[idx]],
                                [discreteAnnotationsEndSeconds[idx], discreteAnnotationsArousalValues[idx]]])

                pl.plot(exc[:, 0], exc[:, 1], 'k', lw=4)

            pl.plot(np.asanyarray(continuousAnnotationsSeconds), np.asanyarray(continuousAnnotationsArousalValues), 'b',
                    lw=1, label='mean')
            pl.fill_between(np.asanyarray(continuousAnnotationsSeconds),
                            np.asanyarray(continuousAnnotationsArousalValues) - 1.96 * np.asanyarray(
                                continuousAnnotationsArousalStd),
                            np.asanyarray(continuousAnnotationsArousalValues) + 1.96 * np.asanyarray(
                                continuousAnnotationsArousalStd),
                            color='grey', alpha='0.3', label='95% confidence interval')
            pl.xlim([0, max(max(continuousAnnotationsSeconds), max(discreteAnnotationsEndSeconds))])
            pl.ylim([-1, 1])
            pl.ylabel('Arousal')

            pl.subplot2grid((4, 2), (2, 0), colspan=2, rowspan=2, aspect=1.0)
            nbGrid = max(1, int(len(continuousAnnotationsSeconds) / 10 + 0.5))
            for i in range(nbGrid):
                firstIdx = max(0, int(len(continuousAnnotationsSeconds) * i / nbGrid) - 1)
                lastIdx = int(len(continuousAnnotationsSeconds) * (i + 1) / nbGrid)
                if (nbGrid > 1):
                    alphaValue = min(1.0, float(0.2 + i * 0.8 / (nbGrid - 1)))
                else:
                    alphaValue = 1.0
                pl.plot(continuousAnnotationsArousalValues[firstIdx:lastIdx],
                        continuousAnnotationsValenceValues[firstIdx:lastIdx], 'b-', alpha=alphaValue)

            pl.scatter(discreteAnnotationsArousalValues, discreteAnnotationsValenceValues, s=20, c='k', marker='o')

            pl.scatter([np.mean(discreteAnnotationsArousalValues)], [np.mean(discreteAnnotationsValenceValues)], s=50,
                       c='r', marker='o')
            pl.scatter([np.mean(continuousAnnotationsArousalValues)], [np.mean(continuousAnnotationsValenceValues)],
                       s=50, c='c', marker='o')
            pl.xlim([-1, 1])
            pl.xlabel('Arousal')
            pl.ylim([-1, 1])
            pl.ylabel('Valence')

            if not os.path.exists(outputFolder):
                os.makedirs(outputFolder)

            # pl.savefig(outputFolder + movieName + '.pdf', bbox_inches='tight')
            pl.savefig(outputFolder + movieName + '.png', bbox_inches='tight')
            pl.close(fig)

    print('\nGlobal Statistics for continuous annotations')
    print('Arousal:')
    print('\tMin: ' + str(minContinuousArousal[1]) + ' (' + minContinuousArousal[0] + ')')
    print('\tMax: ' + str(maxContinuousArousal[1]) + ' (' + maxContinuousArousal[0] + ')')
    print('\tMean: ' + str(meanContinuousArousal / len(movieNames)))
    print('\tMean dynamic: ' + str(meanDynamicContinuousArousal / len(movieNames)))
    print('Valence:')
    print('\tMin: ' + str(minContinuousValence[1]) + ' (' + minContinuousValence[0] + ')')
    print('\tMax: ' + str(maxContinuousValence[1]) + ' (' + maxContinuousValence[0] + ')')
    print('\tMean: ' + str(meanContinuousValence / len(movieNames)))
    print('\tMean dynamic: ' + str(meanDynamicContinuousValence / len(movieNames)))