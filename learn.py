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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


from threading import Thread

from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Manager

import pdb
import math
import time
import io
import datetime
from sklearn import ensemble

# our code
import features

#def _pickle_method(method):
#    func_name = method.im_func.__name__
#    obj = method.im_self
#    cls = method.im_class
#    return _unpickle_method, (func_name, obj, cls)
#
#def _unpickle_method(func_name, obj, cls):
#    for cls in cls.mro():
#        try:
#            func = cls.__dict__[func_name]
#        except KeyError:
#            pass
#        else:
#            break
#    return func.__get__(obj, cls)
#
#import copy_reg
#import types
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def test_prediction_alg(n_estimators=60):
    tr_d = load_data(True)
    te_d = load_data(False)
    m = Model(tr_d[:-10000])

    m.train(n_estimators=n_estimators)
    predictions = m.predict(data = tr_d[-10000:])
    
    training_set_error = predictions.training_set_error(tr_d[-10000:])

    print 'training set error:'
    print training_set_error

    #predictions.write()
    e = tr_d[-10000:]
    e['vote_p'] = predictions.vote_p
    e['view_p'] = predictions.view_p
    e['comment_p'] = predictions.comment_p
    return (training_set_error,m, predictions, e)

def identify_dupes(data_set = None):
    pass

def make_predictions2(n_estimators=20):
    tr_d = load_data(True)
    te_d = load_data(False)
    m = Model(tr_d)
    m.train(n_estimators=n_estimators)
    predictions = m.predict(data = te_d)

#    #horrible hack:
#    for i,comment in enumerate(predictions.comment_p):
#        if comment>5:
#            predictions.comment_p[i] = 0

    # probable error if this violated:
    assert(np.max(predictions.comment_p)<5) 
    predictions.correct_means()
    assert(np.max(predictions.comment_p)<5)
    predictions.vote_p = np.maximum(predictions.vote_p, 1)
    predictions.write(te_d)
    return (m, predictions)

#TODO - This fn is confusing IMO,
def load_data(training_set,after_row = 160000):
    ''' Loads training or test data. '''
    if training_set:
        d = pandas.read_csv("data/train.csv")
#        d = pandas.read_csv("data/train_city_fixed.csv")
        return d[after_row:]
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
        #self.shape = [len(set(d[self.str]))]
        self.shape = [np.max(d[self.str])+1]
        return self
    def transform(self, d):
        return np.array([d[self.str]])
    def __repr__(self):
        return self.str

chicago_replacement_dic = {'Abandoned Vehicle': 'abandoned_vehicle',
                       'Alley Light Out': 'street_light',
                       'Building Violation': 'building_violation',
                       'Graffiti Removal': 'graffiti',
                       'Pavement Cave-In Survey': 'pavement_survey',
                       'Pothole in Street': 'pothole',
                       'Restaurant Complaint': 'restaurant',
                       'Rodent Baiting / Rat Complaint': 'rodents',
                       'Sanitation Code Violation': 'sanitation',
                       'Street Cut Complaints': 'street_cut',
                       'Street Light 1 / Out': 'street_light',
                       'Street Lights All / Out': 'street_light',
                       'Traffic Signal Out': 'street_signal',
                       'Tree Debris': 'tree'}


def chicago_fix(d):
    for (i, (l, so, su)) in enumerate(zip(d.latitude, d.source, d.summary)):
        if l > 41.5:
            if so == "remote_api_created":
                try:
                    d.tag_type.values[i] = chicago_replacement_dic[su]
                except:
                    pdb.set_trace()
    return d

class Model(object):
    def __init__(self, training_data = None, test_data = None):
        if training_data is None:
            self.tr_d = load_data(training_set = True)
        else:
            self.tr_d = training_data
        if test_data is None:
            self.te_d = load_data(training_set = False)
        else:
            self.te_d = test_data
        self.enc = None
        self.beast_encoder = None
        self.km = None
        #self.te_d = chicago_fix(self.te_d)

    def __make_features__(self, d):
        weekday = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').weekday()
        hour = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').hour
        day_sixth = lambda timestr : hour(timestr) // 3

        if self.km is None:
            (self.km, clusters) = features.neighbourhoods(d)
        else:
            clusters = self.km.predict(d[['latitude','longitude']])
        
        #Three loops below need to be a fn:
        #build a dict of summary frequencies
        summary_freq_dict = {}
        for s in d.summary:
            if s not in summary_freq_dict:
                summary_freq_dict[s] = (d.summary == s).sum()

        #build a dict of tag frequencies
        tag_freq_dict = {}
        for s in d.tag_type:
            if s not in tag_freq_dict:
                tag_freq_dict[s] = (d.tag_type == s).sum()

        #build a dict of tag frequencies
        source_freq_dict = {}
        for s in d.source:
            if s not in source_freq_dict:
                source_freq_dict[s] = (d.source == s).sum()



        feature_dic = {
            'weekday': map(weekday,d.created_time.values), # 7

            'source' : features.feature_to_int(d.source.values, # 9 
                                               category_dict =\
                                               self.s_d),

            'tag_type' : features.feature_to_int(d.tag_type.values, # 43
                                                 category_dict = self.t_d),

            'summary' : features.feature_to_int(d.summary.values,  #9 
                                                category_dict = self.summary_d,),

            'description' : map(int, d.description > 0),
            'city': features.city_feature(d), #clusters #10
            'day_sixth': map(day_sixth,d.created_time.values), # 4
            'naive_nlp': map(features.naive_nlp,d.summary.values),
            'naive_nlp_description': map(features.naive_nlp,d.description.values),
            'summary_length':map(features.string_length,d.summary),
            'description_length':map(features.string_length,d.description),            
            #            #huge number of features

            'dense_neighbourhood':features.dense_neighbourhood(d),
            'angry_post':(map(features.angry_post,d.summary.values)),
            'angry_description':(map(features.angry_post,d.description.values)),



        }
#            'summary_bag_of_words':features.summary_bag_of_words(d)


        for f in feature_dic:
            pass
            #print(f)
            #print(set(feature_dic[f]))

        for a in feature_dic.values():
            if not (len(a) == len(d.id.values)):
                pdb.set_trace()
        
        if self.beast_encoder is None:
            be_pw = F('tag_type') * F('source') + \
                     F('tag_type') * F('city') +\
                     F('tag_type') * F('weekday') +\
                     F('source') * F('city') + \
                     F('source') * F('weekday') +\
                     F('city') * F('weekday') +\
                     F('source') * F('day_sixth') +\
                     F('tag_type') * F('day_sixth') +\
                     F('city') * F('day_sixth') +\
                     F('day_sixth') * F('weekday')

            be_city_focus = F('city') * (\
                                         F('weekday') +\
                                         F('description') +\
                                         F('source') * F('tag_type') + \
                                         F('source') * F('description') \
                                         + F('tag_type') * F('description')\
                                         )

            be_small_niche = (F('tag_type') * F('source') * F('city'))

            be_linear = F('tag_type') + F('source') + F('city') +\
                        F('angry_post') + F('day_sixth') +F('summary_length') + F('description_length') + F('naive_nlp') + F('naive_nlp_description') +\
F('weekday') +F('angry_description') +F('description') + F('summary') #line found to have poor predictive capacity:






            self.beast_encoder = be_linear
            self.beast_encoder.fit(feature_dic)

        int_features = self.beast_encoder.transform(feature_dic).transpose()
        
        print("int_features: "+str(int_features.shape))
        
        if self.enc is None:

            self.vectorizer = TfidfVectorizer(min_df=1)
            corpus = d.summary.values
            X = self.vectorizer.fit_transform(corpus)
            dense_X = X.toarray()

            self.description_vectorizer = TfidfVectorizer(min_df=1)
            description_corpus = map(str,d.description.values)
            X_description = self.description_vectorizer.fit_transform(description_corpus)
            dense_X_description = X_description.toarray()


            print 'dense x shape:'
            print np.shape(dense_X)

            print 'description dense x shape:'
            print np.shape(dense_X_description)

            self.enc = sklearn.preprocessing.OneHotEncoder(
                                            n_values = self.beast_encoder.shape
                                               )

            encoded_features = self.enc.fit_transform(int_features).todense()
            


            print("Encoded feature shape: "+str(encoded_features.shape))
            print encoded_features
            print type(encoded_features)


            print 'encoded features shape before:'
            print np.shape(encoded_features)


            encoded_features = np.concatenate((encoded_features,dense_X),axis=1)
            encoded_features = np.concatenate((encoded_features,dense_X_description),axis=1)


            print 'encoded features shape after:'
            print np.shape(encoded_features)


        else:
            print 'now computing the rest'
            corpus = d.summary.values
            dense_X = self.vectorizer.transform(corpus).toarray()

            encoded_features = self.enc.transform(int_features).todense()

            description_corpus = map(str,d.description.values)

            dense_X_description = self.description_vectorizer.transform(description_corpus).toarray()

            print 'encoded features shape before:'
            print np.shape(encoded_features)
            encoded_features = np.concatenate((encoded_features,dense_X),axis=1)
            encoded_features = np.concatenate((encoded_features,dense_X_description),axis=1)

            print 'encoded features shape after:'
            print np.shape(encoded_features)

            print np.max(encoded_features)

        #at this point we are wanting to add the BOW features


        return encoded_features

    def train(self,n_estimators=65):
        """
        Train the model from the training set.

        Currently using SGDRegressor
        """

        #this stage is confusing, s_d needs to work for
        #__make_features__ to work properly
        self.summary_d = features.make_category_dict(self.tr_d.summary.values,threshold=10)

        self.s_d = features.make_category_dict(self.tr_d.source.values)
        self.t_d = features.make_category_dict(self.tr_d.tag_type.values)

        #return the encoded set of features
        tr_features = self.__make_features__(self.tr_d)

        self.regressors = [0,0,0]#initialize

        manager = Manager()
        regressor_list = manager.list(range(3))

        start = time.time()

        #these three functions are independent, should be executed independently

        p_com = Process(target=self.fit_comments, args=(tr_features,regressor_list,))
        p_com.start()

        p_viw = Process(target=self.fit_views, args=(tr_features,regressor_list,))
        p_viw.start()

        p_vot = Process(target=self.fit_votes, args=(tr_features,regressor_list,))
        p_vot.start()

#        self.fit_comments(tr_features)
#        self.fit_views(tr_features)
#        self.fit_votes(tr_features)

        p_com.join()
        p_viw.join()
        p_vot.join()

        self.regressors = regressor_list

#        self.regressors=[p_com.get(),p_viw.get(),p_vot.get()]

#        pool = Pool(processes=3)
#        pool_comments = pool.apply_async(self.fit_comments, [tr_features])
#        pool_views = pool.apply_async(self.fit_views, [tr_features])
#        pool_votes = pool.apply_async(self.fit_votes, [tr_features])#

#        pool.close()
#        pool.join()

#        self.regressors=[pool_comments.get(),pool_views.get(),pool_votes.get()]



    def fit_votes(self,tr_features,regressor_list):
        print 'Fitting num_votes'
        start = time.time()

        regressor = ensemble.GradientBoostingRegressor(n_estimators=20,
                                                       learning_rate=0.1,
                                                       max_depth=6,
                                                       verbose=0)

#        regressor = ensemble.RandomForestRegressor(n_estimators=20,
#                                                   max_depth=6,
#                                                   verbose=0)

#3
        regressor = linear_model.SGDRegressor(n_iter=35,
                                              alpha=0.0001,
                                              shuffle=True,
                                              power_t=0.15,)



        regressor.fit(tr_features, tog(self.tr_d['num_votes'].values))
        
        
        self.regressors[2] = regressor
        regressor_list[2] = regressor
        print 'Elapsed time %f' %(time.time() - start)

        return regressor

    def fit_views(self,tr_features,regressor_list):
        print 'Fitting num_views'
        start = time.time()

        regressor = ensemble.GradientBoostingRegressor(n_estimators=20,
                                                       learning_rate=0.1,
                                                       max_depth=6,
                                                       verbose=0)


#        regressor = ensemble.RandomForestRegressor(n_estimators=60,
#                                                   max_depth=6,
#                                                   verbose=0)

#1
        regressor = linear_model.SGDRegressor(n_iter=35,
                                              alpha=0.0001,
                                              power_t=0.15,
                                              shuffle = True)

        regressor.fit(tr_features, tog(self.tr_d['num_views'].values))

        self.regressors[1] = regressor
        regressor_list[1] = regressor
        print 'Elapsed time %f' %(time.time() - start)

        return regressor

    def fit_comments(self,tr_features,regressor_list):
        print 'Fitting num_comments'
        start = time.time()

#        regressor = ensemble.GradientBoostingRegressor(n_estimators=40,
#                                                   learning_rate=0.1,
#                                                   max_depth=4,
#                                                   verbose=0)

#        regressor = ensemble.RandomForestRegressor(n_estimators=20,
#                                                   max_depth=4,
#                                                   verbose=0)
#
        regressor = ensemble.GradientBoostingRegressor(n_estimators=20,
                                                       learning_rate=0.1,
                                                       max_depth=6,
                                                       verbose=0)

#2
        regressor = linear_model.SGDRegressor(n_iter=35,
                                              alpha=0.0001,
                                              power_t=0.15,
                                              shuffle = True)

        regressor.fit(tr_features, tog(self.tr_d['num_comments'].values))

        self.regressors[0] = regressor
        regressor_list[0] = regressor
        print 'Elapsed time %f' %(time.time() - start)

        return regressor

    def predict(self, data = None, training_set = True):
        if data is None:
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

        print 'num_comments rms error: %f' % rms(e1)
        print 'num_views rms error: %f' % rms(e2)
        print 'num_votes rms error: %f' % rms(e3)

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

    def write(self, d, file = "predictions.csv"):
        assert(self.corrected)
        assert(np.min(self.vote_p)>= 1)
        #pdb.set_trace()
        id = ['id']
        num_views = ['num_views']
        num_votes = ['num_votes']
        num_comments = ['num_comments']

        ids = d['id'].values # np.concatenate([id, d['id'].values.astype(dtype = 'S')])
        assert(np.max(self.comment_p)<5)
        comments = self.comment_p# np.concatenate([num_comments, self.comment_p])
        assert(np.max(comments)<5)
        views = self.view_p# np.concatenate([num_views, self.view_p])
        votes = self.vote_p# np.concatenate([num_votes, self.vote_p])

        with open(file,'w') as handle:
            for (id, comment, view, vote) in zip(ids, comments, views, votes):
                assert(comment < 5)
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

if __name__ == "__main__":
    #make_predictions2()
    while True:
        test_prediction_alg()
