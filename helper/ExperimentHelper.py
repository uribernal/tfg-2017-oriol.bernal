import os
import json
import datetime
import xlsxwriter
import time
import numpy as np


def get_experiments():
    path = 'experiments.json'
    if os.path.isfile(path):
        with open(path) as data_file:
            data = json.load(data_file)
    else:
        data = {}

    return data


def get_actual_experiment_id():
    experiments = get_experiments()
    return len(experiments.keys())


def get_elapsed_time(elapsed):
    hours = 0
    minutes = 0
    if elapsed / 3600 >= 1:
        hours = int(elapsed / 3600)
        minutes = int((elapsed % 3600) / 60)
    elif elapsed / 60 >= 1:
        minutes = int(elapsed / 60)
    return '{0:02d}h{1:02d}min'.format(hours, minutes)


class Experiment:
    'Common base class for all Experiments'
    json_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/logs/experiments.json'
    xls_path = '/home/uribernal/PycharmProjects/tfg-2017-oriol.bernal/results/logs/experiments.xls'
    experiment_id = None
    date = None
    start = None
    elapsed = None
    lstm_cells = None
    optimizer = None
    batch_size = None
    timesteps = None
    dropout = None
    data_split = None
    num_epochs = None
    scores = None

    def __init__(self, num_epochs, lstm_cells, optimizer, batch_size, timesteps, dropout, data_split):
        # Get last experiment
        Experiment.experiment_id = get_actual_experiment_id()

        # Compute date
        Experiment.date = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

        Experiment.start = time.time()

        Experiment.lstm_cells = lstm_cells
        Experiment.optimizer = optimizer
        Experiment.batch_size = batch_size
        Experiment.timesteps = timesteps
        Experiment.dropout = dropout
        Experiment.data_split = data_split
        Experiment.num_epochs = num_epochs

        print('Experiment: {}'.format(Experiment.experiment_id))

    def save_results(self, scores):
        Experiment.elapsed = get_elapsed_time(time.time() - Experiment.start )
        Experiment.scores = scores
        #Experiment.result2 = result2

        # Save Experiment
        Experiment.save_json(self)
        Experiment.save_xls(self)

    def save_json(self):
        # Get experiments
        experiments = get_experiments()

        # Update experiments
        experiments[str(Experiment.experiment_id)] = {
            'date': Experiment.date,
            'elapsed': Experiment.elapsed,
            'lstm_cells': Experiment.lstm_cells,
            'optimizer': Experiment.optimizer,
            'batch_size': Experiment.batch_size,
            'timesteps': Experiment.timesteps,
            'dropout': Experiment.dropout,
            'data_split': Experiment.data_split,
            'num_epochs': Experiment.num_epochs,
            'MSE': Experiment.scores[0],
            'PCC': Experiment.scores[1]}

        # Update JSON
        s = json.dumps(experiments)
        with open(self.json_path, 'w') as f:
            f.write(s)

    def save_xls(self):
        items = ['date', 'elapsed', 'num_epochs', 'lstm_cells', 'optimizer', 'batch_size', 'timesteps', 'dropout', 'data_split', 'MSE', 'PCC']
        keys = np.arange(get_actual_experiment_id())

        # Get experiments
        experiments = get_experiments()

        # Update XLS
        workbook = xlsxwriter.Workbook(self.xls_path)
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'experiment_id')

        col = 0
        for key in keys:
            key = str(key)
            row = 0
            worksheet.write(col + 1, row, key)
            for item in items:
                # If never created, first col with names
                if col == 0:
                    worksheet.write(col, row + 1, item)
                # Write experiment
                worksheet.write(col + 1, row + 1, experiments[key][item])
                row += 1
            col += 1

        workbook.close()
