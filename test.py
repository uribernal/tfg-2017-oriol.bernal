import numpy as np
import matplotlib.pyplot as plt
'''
MSE_AROUSAL = [2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.38815316, 2.56945092, 2.56945092, 2.56945092, 2.56945092, 2.56945092, 2.56945092, 2.56945092, 2.56945092]
MSE_VALENCE = [0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.34387637, 0.33470693, 0.33470693, 0.33470693, 0.33470693, 0.33470693, 0.33470693, 0.33470693, 0.33470693]
min_a = np.argmin(MSE_AROUSAL) #0
min_v = np.argmin(MSE_VALENCE) #32

a,b = (min_a, min(MSE_AROUSAL))

plt.figure(1)
plt.subplot(211)
plt.plot(MSE_AROUSAL)
plt.plot(min_a, min(MSE_AROUSAL),'ro')


plt.figure(1)
plt.subplot(212)
plt.plot(MSE_VALENCE)
plt.plot(min_v, min(MSE_VALENCE),'ro')

plt.show()

'''

from fine_tuning.albertoswork import video_extraxting_fps2 as DB

v = np.array([])
a = np.array([])

movies = DB.get_movies_names()
for movie in movies:
    valence, arousal = DB.get_labels(movie)
    v = np.append(v, valence)
    a = np.append(a, arousal)

print(v.shape)
print(a.shape)
print('\n')

max_v = np.max(v)
max_a = np.max(a)
min_v = np.min(v)
min_a = np.min(a)

print(max_v)
print(max_a)
print(min_v)
print(min_a)