import numpy as np
import matplotlib.pyplot as plt
import h5py

def save_plots(iteration, train_loss, validation_loss, experiment_id):

    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'Train2017_emotion_classification_{experiment_id}_e{epoch:03}.png'

    # Show plots
    x = np.arange(len(validation_loss))
    fig = plt.figure(1)
    fig.suptitle('LOSS', fontsize=14, fontweight='bold')

    # LOSS: TRAINING vs VALIDATION
    plt.plot(x, train_loss, '--', linewidth=2, label='train')
    plt.plot(x, validation_loss, label='validation')
    plt.legend(loc='upper right')

    # MIN
    val, idx = min((val, idx) for (idx, val) in enumerate(validation_loss))
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val-0.01),
                arrowprops=dict(facecolor='black', shrink=0.0005))

    plt.savefig(path+file.format(experiment_id=experiment_id, epoch=iteration), dpi=fig.dpi)
    plt.close()


