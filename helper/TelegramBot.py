"""
This assistant sends messages and images to a chat in telegram.
Used to advise when the programs finish and to send results directly
to mobile phones.
"""

import telegram
import numpy as np
import matplotlib.pyplot as plt
import json
import xlsxwriter
import os
import datetime
import time


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


def send_results(image_path=None, scores=None):
    if image_path is not None:
        send_image(image_path)
    if scores is not None:
        send_message('Valence MSE = {0}\n'.format(scores[0]) +
                     'Arousal MSE = {0}\n'.format(scores[1]) +
                     'Valence PCC = {0}\n'.format(scores[2]) +
                     'Arousal PCC = {0}\n'.format(scores[3]))


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


def get_experiments():
    path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/log.json'
    if os.path.isfile(path):
        with open(path) as data_file:
            data = json.load(data_file)
    else:
        data = {}

    return data


def save_experiment(optimizer, batchsize, timesteps, dropout, n_folds, lr, p1, p2, input_features, layers, cells, scores):
    # Compute date
    date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    # Get experiments
    experiments = get_experiments()

    # Get last experiment
    experiment_id = len(experiments.keys())
    # Update experiments
    experiments[str(experiment_id)] = {
        'log': date,
        'optimizer': str(optimizer.__class__)[25:-2],
        'batch_size': batchsize,
        'timesteps': timesteps,
        'dropout': dropout,
        'n_folds': n_folds,
        'lstm_layers': layers,
        'lstm_cells': cells,
        'starting_lr': lr,
        'lr_patience': p1,
        'stop_patience': p2,
        'input': input_features,
        'MSE valence': scores[0],
        'MSE arousal': scores[1],
        'PCC valence': scores[2],
        'PCC arousal': scores[3]}

    # Update JSON
    s = json.dumps(experiments)
    with open('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/log.json', 'w') as f:
        f.write(s)

    # Update XLS
    workbook = xlsxwriter.Workbook('/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/log.xls')
    worksheet = workbook.add_worksheet()
    d = experiments
    worksheet.write(0, 0, 'experiment_id')

    # col = d.keys()
    col = 0
    for key in d.keys():
        row = 0
        worksheet.write(col + 1, row, key)
        for item in d[key]:
            if col == 0:
                worksheet.write(col, row + 1, item)
            worksheet.write(col + 1, row + 1, d[key][item])
            row += 1
        col += 1

    workbook.close()


def get_actual_experiment_id():
    experiments = get_experiments()
    return len(experiments.keys())


def start_experiment():
    start = time.time()
    experiment_id = get_actual_experiment_id()
    send_message('Starting experiment {0}...'.format(experiment_id))
    return start, experiment_id


def end_experiment(start, image_path, scores):
    end = time.time()
    send_elapsed_time(end-start)
    send_results(image_path, scores)


