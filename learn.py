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
#from features import * #best to avoid this
import features

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def make_predictions():
    m = Model()
    m.train()
    predictions = m.predict()
    return (m, predictions)

#TODO - This fn is confusing IMO,
def load_data(training_set):
    ''' Loads training or test data. '''
    if training_set:
        return pandas.read_csv("data/train.csv")
    else:
        return pandas.read_csv("data/test.csv")

def tog(x):
    return np.log(x + 1)

def untog(x):
    return np.exp(x) - 1

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

class Model(object):
    def __init__(self, training_data = None, test_data = None):
        if training_data is None:
            self.tr_d = load_data(training_set = True)
        if test_data is None:
            self.te_d = load_data(training_set = False)
        self.enc = None

    def __make_features__(self, d):

        import datetime
        weekday = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').weekday()

        int_features = np.zeros((len(d.id), 5))
        int_features[:,0] = features.feature_to_int(d.source.values, category_dict = self.s_d) 
        # 9 values in training set
        int_features[:,1] = features.feature_to_int(self.tr_d.tag_type.values, category_dict = self.t_d)

        int_features[:,2] = features.city_feature(d) 
        int_features[:,3] = map(int, d.description > 0)
        int_features[:,4] = map(weekday,d.created_time.values) #test
        
        # 43 values in training set
        if self.enc is None:
            self.enc = sklearn.preprocessing.OneHotEncoder()
            encoded_features = self.enc.fit_transform(int_features).todense()
        else:
            encoded_features = self.enc.transform(int_features).todense()
        return encoded_features

    def train(self):
        """
        Train the model from the training set.

        Currently using SGDRegressor
        """

        #this stage is confusing, s_d needs to work for
        #__make_features__ to work properly
        self.s_d = features.make_category_dict(self.tr_d.source.values)
        self.t_d = features.make_category_dict(self.tr_d.tag_type.values)

        #return the encoded set of features
        tr_features = self.__make_features__(self.tr_d)

        self.regressors = []

        start = time.time()
        for col_name in cols_to_predict:
            r = SGDRegressor(loss = "squared_loss",
                             n_iter = 10,
                             alpha = 0,
                             power_t = 0.1,
                             shuffle = True)

            r.fit(tr_features, tog(self.tr_d[col_name].values))

            self.regressors.append(r)

            print(time.time() - start)

    def predict(self, training_set = False):
        
        if training_set:
            data = self.tr_d
        else:
            data = self.te_d
        
        data = self.tr_d if training_set else self.te_d

        features = self.__make_features__(data) 
        prediction_arr = []
        for r in self.regressors:
            prediction_arr.append(untog(r.predict(features)))
            prediction_arr[-1]  = np.maximum(prediction_arr[-1], 0)
        predictions = Predictions(prediction_arr[0], prediction_arr[1], prediction_arr[2])
        if not training_set:
            predictions.correct_means()
        return predictions

class Predictions(object):
    ''' Represents a set of test predictions. '''
    def __init__(self, comment_p, view_p, vote_p):
        self.comment_p = comment_p
        self.view_p = view_p
        self.vote_p = vote_p

    def training_set_error(self,training_data):
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
        self.comment_p = Predictions.__set_tog_mean__(self.comment_p, log_mean_comments) 
        self.view_p = Predictions.__set_tog_mean__(self.view_p, log_mean_views)
        self.vote_p = Predictions.__set_tog_mean__(self.vote_p, log_mean_votes)
        self.corrected = True

    @staticmethod
    def __set_tog_mean__(arr, m):
        arr = np.array(arr) # solves an annoying bug
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
        assert(np.min(self.vote_p)>= 1)

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
