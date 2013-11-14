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
#    regressors = [ensemble.GradientBoostingRegressor(n_estimators=i) for i in range(80,120)]
#    regressors = [ensemble.GradientBoostingRegressor(n_estimators=100)]

#    r = ensemble.GradientBoostingRegressor(n_estimators=100,
#                                           learning_rate=0.1,
#                                           max_depth=3,
#                                           verbose=0)

#    r = ensemble.GradientBoostingRegressor(n_estimators=100,
#                                           learning_rate=0.1,
#                                           max_depth=3,
#                                           verbose=0)

#    data = learn.test_prediction_alg(n_estimators=1)



#    prediction = lambda num_estimators : learn.test_prediction_alg(n_estimators=num_estimators)
#    data = prediction(2)
    data = pool.map(prediction,range(1,31))
    data = np.array(data)
    scores =  data[:,0]

    plt.plot(scores)
    plt.show()

scan()
