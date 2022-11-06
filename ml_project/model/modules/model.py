from preprocessing import custom_transformer
from entities import FeatureParams, BadModel

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd


def get_model(data_config: dict, df: pd.DataFrame, target: pd.DataFrame,
              model_name: str = 'svc', best_finding: bool = False):
    model_dict = {'svc': SVC(), 'knn': KNeighborsClassifier(),
                  'linear': LinearRegression(), 'naive': GaussianNB()}
    if not best_finding:
        if model_name.lower() not in model_dict:
            raise BadModel(model_dict.keys())

        res_model = Pipeline([('transformer', custom_transformer(
            FeatureParams(numerical=data_config['numerical'],
                          str_features=data_config['categorial']))),
                              (model_name, model_dict[model_name])])
    else:
        best_score = -1
        res_model = None
        for model, name in model_dict.items():
            X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                                random_state=0)
            pipe = Pipeline([('transformer', custom_transformer(
                FeatureParams(numerical=data_config['numerical'],
                              str_features=data_config['categorial']))),
                             (name, model)])
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            res_model = model if score > best_score else res_model

    res_model.fit(df, target)
    return res_model
