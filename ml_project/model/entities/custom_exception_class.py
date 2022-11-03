class BadMetric(Exception):
    """Exception raised for errors in metric function.

    """
    def __init__(self, metric_list=('accuracy', 'precision',
                                    'recall', 'roc_auc')):

        super().__init__(f'A bad metric was used. '
                         f'A metric was expected :{metric_list}')


class BadModel(Exception):
    """Exception raised for errors in model function

    """
    def __init__(self, model_list):
        super().__init__(f'A bad model was used. '
                         f'A model was expected :{model_list}')
