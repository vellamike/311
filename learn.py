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
import datetime

# our code
#from features import * #best to avoid this
import features

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def make_predictions():
    m = Model()
    m.train()
    predictions = m.predict(training_set = True)
    print(predictions.training_set_error(load_data(True)))
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

class BeastEncoder:
    def __add__(self, other):
        return AddEncoder(self, other)
    def __mul__(self, other):
        return MulEncoder(self, other)
    def fit(self, d):
        return self
    def transform(self, d):
        return self

class AddEncoder(BeastEncoder):
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def fit(self, d):
        self.first.fit(d)
        self.second.fit(d)
        self.shape = self.first.shape + self.second.shape
        return self
    def transform(self, d):
        return np.vstack((self.first.transform(d), self.second.transform(d)))
    def __repr__(self):
        return "(" + str(self.first) + " + " + str(self.second) + ")"

class MulEncoder(BeastEncoder):
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def fit(self, d):
        self.first.fit(d)
        self.second.fit(d)
        self.shape = []
        for v in self.first.shape:
            for w in self.second.shape:
                self.shape.append(v * w)
        return self
    def transform(self, d):
        tf = self.first.transform(d)
        ts = self.second.transform(d)
        cols = []
        for (cf, v) in zip(tf, self.first.shape):
            for (cs, w) in zip(ts, self.second.shape):
                cols.append(v * cs + cf)
        return np.vstack(cols)
    def __repr__(self):
        return str(self.first) + " * " + str(self.second)

class F(BeastEncoder):
    def __init__(self, str):
        self.str = str
    def fit(self, d):
        self.shape = [len(set(d[self.str]))]
        return self
    def transform(self, d):
        return np.array([d[self.str]])
    def __repr__(self):
        return self.str



class Model(object):
    def __init__(self, training_data = None, test_data = None):
        if training_data is None:
            self.tr_d = load_data(training_set = True)
        if test_data is None:
            self.te_d = load_data(training_set = False)
        self.enc = None
        self.beast_encoder = None

    def __make_features__(self, d):
        
        weekday = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').weekday()
        
        
        feature_dic = {
            'weekday': map(weekday,d.created_time.values), # 7
            'source' : features.feature_to_int(d.source.values, # 9 
                                               category_dict =\
                                               self.s_d),
            'tag_type' : features.feature_to_int(d.tag_type.values, # 43
                                                 category_dict = self.t_d),
            'description' : map(int, d.description > 0),
            'city': features.city_feature(d)} 

        for a in feature_dic.values():
            if not (len(a) == len(d.id.values)):
                pdb.set_trace()
        
        if self.beast_encoder is None:
            self.beast_encoder = F('weekday') + F('tag_type') * F('source') * F('city')+\
                                 F('description')
            self.beast_encoder.fit(feature_dic)

        int_features = self.beast_encoder.transform(feature_dic).transpose()
        print("int_features: "+str(int_features.shape))
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
                             alpha = 0.01,
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

        features = self.__make_features__(data) 
        print("features: " + str(features.shape))
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
