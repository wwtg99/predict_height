from sklearn.externals import joblib
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
import os


default_model_dir = 'models'

_models = {
    'bayesian_ridge': BayesianRidge(),
    'linear_regression': LinearRegression(),
    'elastic_net': ElasticNet(),
    'lasso': Lasso(),
    'svr': SVR(kernel='linear'),
    'gbr': GradientBoostingRegressor(n_estimators=300, max_depth=5)
}


def get_model_names():
    """
    Get supported model names.

    :return:
    """
    return list(_models.keys())


def get_models(model_name):
    """
    Get models.

    :param model_name:
    :return:
    """
    if isinstance(model_name, list):
        return dict([(i, _models[i]) for i in model_name if i in _models])
    elif model_name in _models:
        return _models[model_name]
    else:
        return None


def save_model(model, model_name, out_dir=None):
    """
    Save model to file.

    :param model:
    :param model_name:
    :param out_dir:
    :return:
    """
    if not out_dir:
        out_dir = default_model_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    outf = out_dir + os.path.sep + model_name + '.model'
    joblib.dump(model, outf)


def save_models(models, model_names, out_dir=None):
    """
    Save models to out_dir.

    :param models:
    :param model_names:
    :param out_dir:
    :return:
    """
    for i in range(len(model_names)):
        if i >= len(models):
            break
        save_model(models[i], model_names[i], out_dir)


def load_model(model_name, out_dir=None):
    """
    Load saved model from model directory.

    :param model_name:
    :param out_dir:
    :return:
    """
    if not out_dir:
        out_dir = default_model_dir
    outf = out_dir + os.path.sep + model_name + '.model'
    if os.path.exists(outf):
        return load_model_from_file(outf)
    return None


def load_model_from_file(filepath):
    """
    Load saved models from file.

    :param filepath:
    :return:
    """
    return joblib.load(filepath)
