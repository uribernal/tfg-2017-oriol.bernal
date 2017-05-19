"""
This assistant sends messages and images to a chat in telegram.
Used to advise when the programs finish and to send results directly
to mobile phones.
"""

import telegram
import numpy as np
import matplotlib.pyplot as plt


def send_message(message: str):
    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendMessage(chat_id=chat_id, text=message)


def send_image(image_path: str):
    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendPhoto(chat_id=chat_id, photo=open(image_path, 'rb'))


def send_elapsed_time(elapsed: int):
    hours = 0
    minutes = 0
    if elapsed/3600 >= 1:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
    elif elapsed/60 >= 1:
        minutes = int(elapsed / 60)
    send_message('Elapsed Time: {0:02d}h{1:02d}min'.format(hours, minutes))


def save_plots(train_loss, validation_loss, path):

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
    plt.annotate(str(val), xy=(idx, val), xytext=(idx, val - 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.0005))

    plt.savefig(path, dpi=fig.dpi)
    plt.close()
    plt.close()
