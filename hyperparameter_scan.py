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

scan()
