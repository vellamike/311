import numpy as np
import io
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pdb
from numpy import genfromtxt
import math

def load_data(path="data/train_four.csv"):
    """
    Output of this function appears to be a structured array,
    which does cause some headaches/benefits, depending on how you look at it:
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    """
    
    data = np.recfromcsv(path, delimiter = ";", invalid_raise = False)
    return data

def add_features(d):
    d.chars_in_description = map(len,d.description)
    d.chars_in_summary = map(len,d.summary)
    d.log_num_views = map(math.log,d.num_views + 1)

def train(d):
    add_features(d)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
                        'learning_rate': 0.01, 'loss': 'ls'}
    enc = sklearn.preprocessing.OneHotEncoder()



    clf = ensemble.GradientBoostingRegressor(**params)
    features = np.column_stack((d['latitude'],
                               d['longitude'],
                               d.chars_in_description,
                               d.chars_in_summary))
    clf.fit(features, d.log_num_views)
    return clf


def predict(clf):
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    pass

def set_log_mean(predictions):
    log_mean_comments = 0.02665824
    log_mean_views = 0.41264568
    log_mean_votes = 0.80850881
    means = (log_mean_comments,
             log_mean_views,
             log_mean_votes)
    # Assume predictions is a 3 tuple of arrays: first comment predictions,
    # then views, then votes.
    scaled_predictions = []
    for (m, p) in zip(means, predictions):
        scale_factor_lb = 0
        scale_factor_ub = 2
        while scale_factor_ub - scale_factor_lb > 10**(-7):
            guess = (scale_factor_ub + scale_factor_lb) /2.0
            mean = np.mean(np.log(1+ guess * p))
            if mean > m:
                scale_factor_ub = guess
            else:
                scale_factor_lb = guess
        scaled_predictions.append(guess * p)
    return scaled_predictions




