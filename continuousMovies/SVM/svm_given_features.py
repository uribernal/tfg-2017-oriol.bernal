import time

import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error

from continuousMovies import loadFeatures
from continuousMovies.helper import bot

bot.sendMessage("START")

start = time.time()

ids, names, labels = loadFeatures.getLabels()
featuresArousal, np_featureValuesArousal, featuresValence, np_featureValuesValence = loadFeatures.getFeatures()

arousal,arousal_lab = np_featureValuesArousal, labels[:,3]
valence,valence_lab = np_featureValuesValence, labels[:,2]

test_arousal, test_arousal_lab = arousal[-50:,:], arousal_lab[-50:]
test_valence, test_valence_lab = valence[-50:,:], valence_lab[-50:]

train_arousal, train_arousal_lab = arousal[0:-50,:], arousal_lab[0:-50]
train_valence, train_valence_lab = valence[0:-50,:], valence_lab[0:-50]
print (type(train_arousal[0]))
print (type(train_arousal_lab[0]))

gammas = [0.00001, 0.0001, 0.001, 0.01, 0.1]
Cs = [1, 10, 20, 40, 60, 80, 100, 130]
i = 0
mse_arousal = np.zeros(5*8)
mse_valence = np.zeros(5*8)

for gamma in gammas:
    for C in Cs:
        clf_arousal = svm.SVC(gamma=gamma, C=C)
        clf_valence = svm.SVC(gamma=gamma, C=C)
        clf_arousal.fit(train_arousal,train_arousal_lab)
        clf_valence.fit(train_valence,train_valence_lab)
        a = clf_arousal.predict(test_arousal)
        b = clf_valence.predict(test_valence)
        pred_arousal = np.array(a, dtype=float)
        pred_valence = np.array(b, dtype=float)
        lab_arousal = np.array(test_arousal_lab, dtype=float)
        lab_valence = np.array(test_valence_lab, dtype=float)
        mse_arousal[i] = mean_squared_error(pred_arousal, lab_arousal)
        mse_valence[i] = mean_squared_error(pred_valence, lab_valence)
        i+=1

end = time.time()
elapsed = end - start

print("----------------")
print("MSE_AROUSAL: " + str(mse_arousal))
print("MSE_VALENCE: " + str(mse_valence))
print("----------------")
print("Total time: " + str(elapsed))
print("----------------")

bot.sendMessage("FINISHED")
bot.sendMessage("MSE_AROUSAL: " + str(mse_arousal))
bot.sendMessage("MSE_VALENCE: " + str(mse_valence))
bot.sendMessage("Elapsed time:" + str(elapsed))

