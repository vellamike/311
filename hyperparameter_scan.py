import itertools
from sklearn import ensemble
from multiprocessing import Pool
import multiprocessing
import multiprocessing.pool
import learn
from matplotlib import pyplot as plt
import numpy as np
from learn import F
import pdb
from operator import itemgetter

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of
# multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


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


def min_index(array):
    index_of_min = min(enumerate(array), key=itemgetter(1))[0]
    return index_of_min

def multi_test_prediction(encoder,num_trials=8):

    pool = MyPool(num_trials)

    encoder_list = [encoder] * num_trials
    results = pool.map(learn.test_prediction_alg,encoder_list)

    scores = np.array([result[0] for result in results])
    
    overall = scores.T[0].mean()
    comments = scores.T[1].mean()
    views = scores.T[2].mean()
    votes = scores.T[3].mean()

#    print results
#    pdb.set_trace()

    return [overall,comments,views,votes]

def feature_scan(features=None):

    core_features =  [F('tag_type'),
                      F('source'),
                      F('city'),
                      F('angry_post'),                  
                      F('day_sixth'),]

    extra_features =  [F('summary_length'),
                       F('description_length'),
                       ]

    test_features =    [F('summary'),
                        F('angry_description'),
                        F('description'),
                        F('naive_nlp'),
                        F('naive_nlp_description'),
                        F('weekday'),
                       ]

    all_features = core_features + extra_features

    add = lambda x,y : x + y
    naive_linear_encoder = reduce(add,all_features)

    feature_list = test_features

    encoders = []

    for num_features in range(2,len(feature_list)+1):
        for subset in itertools.combinations(feature_list,num_features):
            new_encoder = reduce(add,subset)
            encoders.append(new_encoder)
    
    #number of encoders:
    print len(encoders)

    #now loop through the encoders, computing the mean fitnesses:
    overall_fitness = []
    votes_fitness = []
    comments_fitness = [] 
    views_fitness = []
    
    for encoder in encoders:
#        fitnesses = learn.test_prediction_alg(encoder=encoder)
        fitnesses =multi_test_prediction(encoder)
        errors = fitnesses#[0]
        overall_fitness.append(errors[0])
        comments_fitness.append(errors[1])
        views_fitness.append(errors[2])
        votes_fitness.append(errors[3])

    print encoders
    print overall_fitness

    votes_min_index = min_index(votes_fitness)
    views_min_index = min_index(views_fitness)
    comments_min_index = min_index(comments_fitness)

    print 'best votes encoder:'
    print encoders[votes_min_index]
    
    print 'votes score:'
    print votes_fitness[votes_min_index]
 
    print 'best comments encoder:'
    print encoders[comments_min_index]

    print 'comments score:'
    print comments_fitness[comments_min_index]

    print 'best views encoder:'
    print encoders[views_min_index]

    print 'views score:'
    print views_fitness[views_min_index]

    print overall_fitness
    print votes_fitness
    print views_fitness
    print comments_fitness

feature_scan()
