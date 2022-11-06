import glob
import unittest
from random import randint

import click
import pandas as pd
from sklearn.model_selection import train_test_split

from model.preprocessing import get_config
from model.modules import metric_calc, get_model

from model.entities import BadMetric


class ModelTest(unittest.TestCase):
    @click.option('--conf', default='configs/train_config.yml',
                  help='Path to config file')
    def test_model(self, conf='configs/train_config.yml'):
        config = get_config(conf)['feature_params']

        data = [[randint(40, 80), randint(0, 1), randint(0, 3),
                 randint(100, 180), randint(50, 300), randint(0, 1),
                 randint(0, 2), randint(90, 180), randint(0, 1),
                 randint(0, 9), randint(0, 2), randint(0, 3), randint(0, 2)]
                for _ in range(10000)]
        target = [randint(0, 1) for _ in range(10000)]
        columns = list(config['categorial']) + list(config['numerical'])
        data = pd.DataFrame(data, columns=columns)
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.8, random_state=42,
            shuffle=True)
        model = get_model(config, X_train, y_train)
        self.assertLess(metric_calc(model.predict(X_test), y_test,
                                    'roc_auc'), 0.55)
        self.assertLess(metric_calc(model.predict(X_test), y_test,
                                    'accuracy'), 0.55)
        with self.assertRaises(BadMetric):
            metric_calc(model.predict(X_test), y_test, 'SomeRandomPhrase')

    def test_configs(self):
        config_files = glob.glob('configs/*')
        self.assertListEqual(sorted(config_files),
                             sorted(['configs/train_config.yml',
                                     'configs/eval_config.yml']))


if __name__ == '__main__':
    unittest.main()
