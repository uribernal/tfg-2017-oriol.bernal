#http://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
FFMPEG_BIN = "ffmpeg" # on Linux ans Mac OS
import numpy as np

import subprocess as sp
command = [ FFMPEG_BIN,
            '-i', '/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Spaceman.mp4', #624x352
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
nb_frames=22287
video = np.array([])
# read 420*360*3 bytes (= 1 frame)
for i in range (0, nb_frames):
    raw_image = pipe.stdout.read(352*624*3)
    # transform the byte read into a numpy array
    image =  np.fromstring(raw_image, dtype='uint8')
    image = image.reshape((352,624,3))
    video = np.append(video, image)
    # throw away the data in the pipe's buffer.
    pipe.stdout.flush()

#video2 = np.array(video)
print(type(video))
print(video.size)
print('fin')
import matplotlib.pyplot as plt

plt.figure(3)
plt.imshow(image)
plt.show()


#GET DIMENSIONS OF THE VIDEO
    # ffprobe -v error -show_entries stream=width,height \>   -of default=noprint_wrappers=1 /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Spaceman.mp4
#GET NUMBER OF FRAMES
    # ffprobe -select_streams v -show_streams /home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/LIRIS-ACCEDE-continuous-movies/continuous-movies/Spaceman.mp4