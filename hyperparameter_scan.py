from sklearn import ensemble
from multiprocessing import Pool
import learn
from matplotlib import pyplot as plt
import numpy as np


def prediction(num_estimators):
    """
    multiprocessing won't work with lambdas
    """
    return learn.test_prediction_alg(n_estimators=num_estimators)

def scan(num_processes=64):
    pool = Pool(processes=num_processes)

    data = pool.map(prediction,range(1,31))
    data = np.array(data)
    scores =  data[:,0]

    plt.plot(scores)
    plt.show()


def feature_scan(features=None):

    default_features =  [F('tag_type'),
                         F('source'),
                         F('city'),
                         F('angry_post'),
                         F('day_sixth'),
                         F('summary_length'),
                         F('description_length'),
                         F('naive_nlp'),
                         F('naive_nlp_description'),
                         F('weekday'),
                         F('angry_description'),
                         F('description'),
                         F('summary')]
    
    naive_linear_encoder = sum(default_features)

    learn.test_prediction_alg(encoder=naive_linear_encoder)
