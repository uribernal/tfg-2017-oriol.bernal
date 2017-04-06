import telegram
import time
from tqdm import * #Progress Bar

def sendMessage(message: str):
    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendMessage(chat_id=chat_id, text=message)

def sendImage(experiment_id: int, iteration: int):
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/figures/'
    file = 'FINAL_lstm_emotion_classification_{experiment_id}_e{epoch:03}.png'

    image_path = path + file.format(experiment_id=experiment_id, epoch=iteration)
    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id
    bot.sendPhoto(chat_id=chat_id, photo=open(image_path, 'rb'))

def sendElapsedTime(elapsed: int):
    hours = 0
    minutes = 0
    if elapsed/3600 >= 1:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
    elif elapsed/60 >= 1:
        minutes = int(elapsed / 60)
    sendMessage('Elapsed Time: {0:02d}h{1:02d}min'.format(hours, minutes))


def setProgressBar(i, rg):
    tqdm(range(rg))
