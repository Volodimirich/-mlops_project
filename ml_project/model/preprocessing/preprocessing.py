from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from model.entities import FeatureParams


def custom_transformer(params: FeatureParams) -> ColumnTransformer:
    transfomer_list = []
    if params.str_features:
        transfomer_list.append(('column', OneHotEncoder(),
                                params.str_features))

    if params.numerical:
        transfomer_list.append(('num', StandardScaler(), params.numerical))
    return ColumnTransformer(transfomer_list)
