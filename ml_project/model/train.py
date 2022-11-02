import click
import logging
import pickle
import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from entities.train_params import SplittingParams
from model.preprocessing.data_load import get_dataset, get_config

if not os.path.exists('../logs'):
    os.mkdir('../logs')
logger = logging.getLogger(__name__)
handler = logging.FileHandler('../logs/train.log')
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command()
@click.option('--config', default='configs/eval_config.yml', help='Path to config file')
def train_model(config):
    config = get_config(config)
    path_config = config['path']
    model_config = config['model']

    logger.info(f'Opening the file {config}')
    try:
        X_train, X_test, y_train, y_test = \
            get_dataset(path_config['input_data_path'],
                        SplittingParams(test_size=model_config['test_size'],
                                        random_state=model_config['random_state'],
                                        shuffle=model_config['shuffle']))
        logger.info('get_dataset completed successfully')
    except Exception as err:
        logger.critical(f'Critical error in get_dataset, message - {err}')
        return 1

    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    pipe.fit(X_train, y_train)
    model_score = pipe.score(X_test, y_test)
    logger.info(f'Model SVC get score - {model_score}')

    with open(path_config['output_model_path'], 'wb') as f:
        pickle.dump(pipe, f)
    logger.info(f'Model SVC get score - {model_score}')



if __name__ == '__main__':
    train_model()
