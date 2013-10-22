import numpy as np
import io
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pdb
from numpy import genfromtxt

def load_data(path="data/train_four.csv"):
    """
    Output of this function appears to be a structured array,
    which does cause some headaches/benefits, depending on how you look at it:
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    """
    
    data = np.recfromcsv(path, delimiter = ";", invalid_raise = False)
    return data

def train(data):
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
                        'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)


def predict(clf):
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)
    pass
