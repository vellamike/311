''' Functions to learn, train and predict. ''' 

# maths
import numpy as np
import pandas

# sklearn imports
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor 
import sklearn.preprocessing

import pdb
import math
import time
import io

# our code
import city_extraction

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def load_data(training_set):
    ''' Loads training or test data. '''
    if training:
        return pandas.read_csv("data/train.csv")
    else:
        return pandas.read_csv("data/test.csv")

def tog(x):
    return np.log(x + 1)

def untog(x):
    return np.exp(x) - 1

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

class Model:
    def __init__(self, training_data = None, test_data = None):
        if training_data is None:
            self.tr_d = load_data(training_set = True)
        if test_data is None:
            self.te_d = load_data(training_set = False)

    def __make_features__(self):
        int_features = np.zeros((len(self.tr_d.id), 4))
        int_features[:,0] = feature_to_int(self.tr_d.source.values, category_dict = self.s_d) 
        # 9 values in training set
        int_features[:,1] = feature_to_int(self.tr_d.tag_type.values, category_dict = self.t_d)
        int_features[:,2] = city_feature(self.tr_d)
        int_features[:,3] = map(int, self.tr_d.description > 0)
        # 43 values in training set
        if self.enc is None:
            self.enc = sklearn.preprocessing.OneHotEncoder()
            self.encoded_features =  self.enc.fit_transform(int_features).todense()
        else:
            self.encoded_features = self.enc.transform(int_features).todense()

    def train(self):
        self.s_d = make_category_dict(d.source.values)
        self.t_d = make_category_dict(d.tag_type.values)
        self.make_features()
        self.regressors = []
        start = time.time()
        for col_name in cols_to_predict:
            r = SGDRegressor(loss = "squared_loss", n_iter = 10, alpha = 0, power_t
                            = 0.1, shuffle = True)
            r.fit(features, tog(self.tr_d[col_name].values))
            self.regressors.append(r)
        print(time.time() - start)

    def predict(regressors, s_d, t_d, enc, training_set = False):
        data = self.tr_d if training_set else self.te_d
        (test_features, enc) = make_features(data, source_dic = s_d, tag_dic = t_d,
                                      enc = enc)
        predictions = []
        for r in regressors:
            predictions.append(untog(r.predict(test_features)))
            predictions[-1]  = np.maximum(predictions[-1], 0)
        
        if not training_set:
            predictions.correct_means()
        return predictions

class Predictions:
    ''' Represents a set of test predictions. '''
    def __init__(self, comment_p, view_p, vote_p):
        self.comment_p = comment_p
        self.view_p = view_p
        self.vote_p = vote_p

    def training_set_error(training_data):
        d = training_data 
        e1 = tog(d.num_comments.values) - tog(self.comment_p)
        e2 = tog(d.num_views.values) - tog(self.view_p)
        e3 = tog(d.num_votes.values) - tog(self.vote_p)
        error = rms(np.concatenate([e1,e2,e3]))
        return error

    def correct_means(self):
        log_mean_comments = 0.02665824
        log_mean_views = 0.41264568
        log_mean_votes = 0.80850881
        self.comment_p = set_tog_mean(self.comment_p, log_mean_comments) 
        self.view_p = set_tog_mean(self.view_p, log_mean_views)
        self.vote_p = set_tog_mean(self.vote_p, log_mean_votes)
        self.corrected = True

    def __set_tog_mean__(arr, m):
        scale_factor_lb = 0
        scale_factor_ub = 2
        while scale_factor_ub - scale_factor_lb > 10**(-7):
            guess = (scale_factor_ub + scale_factor_lb) /2.0
            if np.mean(tog(guess * arr)) > m:
                scale_factor_ub = guess
            else:
                scale_factor_lb = guess
        print('''Correction factor: {0}.'''.format(guess))
        return guess * arr

    def write(file = "predictions.csv"):
        assert(self.corrected)

        id = ['id']
        num_views = ['num_views']
        num_votes = ['num_votes']
        num_comments = ['num_comments']

        ids = np.concatenate([id, d['id'].values.astype(dtype = 'S')])
        comments = np.concatenate([num_comments, self.comment_p])
        views = np.concatenate([num_views, self.view_p])
        votes = np.concatenate([num_votes, self.vote_p])

        with open(file,'w') as handle:
            for (id, comment, view, vote) in zip(ids, comments, views, votes):
                handle.write("{0},{1},{2},{3}\n".format(id, view, vote, comment))
    
def make_vw_training_set(d, features):
    with open("data/vowpal_training.vw","w") as handle:
        for (id, tog_views, f) in zip(d.id,
                                      d.tog_num_views, 
                                      features):
            s = "{0} '{1} |".format(tog_views, id) 
            for (i, x) in enumerate(np.array(f)[0]):
                #pdb.set_trace()
                s += "{0}:{1} ".format(i, x)
            handle.write(s + '\n')
