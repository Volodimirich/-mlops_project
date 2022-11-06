import logging
import pickle
import os
import click
from mlflow import log_metric, log_param
from entities import SplittingParams, BadMetric
from modules import metric_calc, get_model
from preprocessing import get_dataset, get_config

if not os.path.exists("logs"):
    os.mkdir("logs")
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/train.log")
formatter = logging.Formatter("%(asctime)s - %(name)s "
                              "" "- %(levelname)s - %(message)s")
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command()
@click.option("--conf", default="configs/train_config.yml",
              help="Path to config file")
def train_model(conf):
    config = get_config(conf)
    path_conf = config["path"]
    mod_conf = config["model"]
    data_conf = config["feature_params"]
    log_param("feature params", data_conf)
    log_param("model params", mod_conf)
    log_param("path params", path_conf)

    logger.info(f"Opening the file {conf}")
    try:
        x_train, x_test, y_train, y_test = get_dataset(
            path_conf["input_data_path"],
            SplittingParams(
                test_size=mod_conf["test_size"],
                random_state=mod_conf["random_state"],
                shuffle=mod_conf["shuffle"],
            ),
            data_conf["target"],
        )
        logger.info("get_dataset completed successfully")
    except BadMetric as err:
        logger.critical(f"Critical error in get_dataset, message - {err}")
        return 1

    try:
        model = get_model(data_conf, x_train, y_train)
        model_score = model.score(x_test, y_test)
        log_metric("model score", model_score)
        logger.info(f"Model get score - {model_score}")

        metric_val = metric_calc(model.predict(x_test), y_test,
                                 mod_conf["metric"])
        log_metric(f'model {mod_conf["metric"]}', metric_val)
        logger.info(f'Model {mod_conf["metric"]}  - {metric_val}')
    except Exception as err:
        logger.critical(f"Critical Exception on training, message - {err}")
        return 1

    with open(path_conf["output_model_path"], "wb") as file:
        pickle.dump(model, file)

    logger.info("Model saved")


if __name__ == "__main__":
    train_model()
