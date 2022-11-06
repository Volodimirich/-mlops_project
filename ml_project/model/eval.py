from datetime import datetime

import click
import logging
import pickle
import os
import numpy as np

from preprocessing import get_eval_data, get_config

if not os.path.exists("logs"):
    os.mkdir("logs")

logger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/eval.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - '
                              '%(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command()
@click.option('--conf', default='configs/eval_config.yml',
              help='Path to config file')
def train_model(conf):
    config = get_config(conf)
    path_config = config['path']
    data_config = config['dataset']
    try:
        with open(path_config['model_path'], 'rb') as f:
            clf = pickle.load(f)
        data = get_eval_data(path_config['input_data_path'])
        if data_config['drop_columns']:
            data.drop(data_config['drop_columns'], axis=1, inplace=True)
        logger.info('Data loaded successfully')
    except Exception as err:
        logger.critical(f'Critical error in get_eval_data, message - {err}')
        return 1

    timestamp = str(datetime.now().date()) + \
                '_' + str(datetime.now().time()).replace(':', '.')

    prediction = clf.predict(data)
    logger.info('Data calculated, saving...')
    np.savetxt(f'{path_config["save_dir"]}'
               f'/result_{timestamp}.csv',
               prediction, delimiter=",")
    logger.info(f'Data saved as result_{timestamp}.csv')


if __name__ == '__main__':
    train_model()
