"""
This assistant sends messages and images to a chat in telegram.
Used to advise when the programs finish and to send results directly
to mobile phones.
"""

import telegram


def send_message(message: str):
    """ Sends a message using telegram API """

    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id

    bot.sendMessage(chat_id=chat_id, text=message)


def send_image(image_path: str):
    """ Sends an image using telegram API """

    bot = telegram.Bot(token='193640162:AAGV3d2H6IAenp3HsdLnuxECL7aLWLGpgmQ')
    updates = bot.getUpdates()
    chat_id = updates[-1].message.chat_id

    bot.sendPhoto(chat_id=chat_id, photo=open(image_path, 'rb'))


def send_elapsed_time(elapsed: int):
    """ Sends the elapsed time using telegram API """

    hours = 0
    minutes = 0
    if elapsed/3600 >= 1:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
    elif elapsed/60 >= 1:
        minutes = int(elapsed / 60)

    send_message('Elapsed Time: {0:02d}h{1:02d}min'.format(hours, minutes))


def send_results(image_path=None, scores=None):
    """ Sends MSE and PCC values, after testing, using telegram API """

    if image_path is not None:
        send_image(image_path)
    if scores is not None:
        send_message('Valence MSE = {0}\n'.format(scores[0]) +
                     'Arousal MSE = {0}\n'.format(scores[1]) +
                     'Valence PCC = {0}\n'.format(scores[2]) +
                     'Arousal PCC = {0}\n'.format(scores[3]))
