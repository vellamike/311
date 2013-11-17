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
from multiprocessing import Pool

import pdb
import math
import time
import io
import datetime
from sklearn import ensemble

# our code
import features

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def test_prediction_alg():
    (tr_d, te_d) = (tr_data(), te_data())
    print("training...")
    m = SubmissionGenerator().train(tr_d[:-10000])
    print("predicting...")
    predictions = m.predict(tr_d[-10000:])
    training_set_error = predictions.training_set_error(tr_d[-10000:])

    print('training set error: {0}'.format(training_set_error))

    e = tr_d[-10000:]
    e['vote_p'] = predictions.vote_p
    e['view_p'] = predictions.view_p
    e['comment_p'] = predictions.comment_p
    return (training_set_error,m, predictions, e)

def make_submission():
    (tr_d, te_d) = (tr_data(), te_data())
    m = SubmissionGenerator().train(tr_d)
    predictions = m.predict(te_d)
    assert(np.max(predictions.comment_p)<5) 
    predictions.correct_means()
    assert(np.max(predictions.comment_p)<5)
    predictions.vote_p = np.maximum(predictions.vote_p, 1)
    predictions.write(te_d)
    return (m, predictions)

def tr_data(after_row = 160000):
    return pandas.read_csv("data/train.csv")[after_row:]

def te_data():
    return pandas.read_csv("data/test.csv")

def tog(x):
    return np.log(x + 1)

def untog(x):
    return np.exp(x) - 1

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

def identify_dupes(data_set = None):
    pass

def add_features(d, s_d, t_d, summary_d):
    weekday = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').weekday()
    hour = lambda timestr : datetime.datetime.strptime(timestr,'%Y-%m-%d %H:%M:%S').hour
    day_sixth = lambda timestr : hour(timestr) // 3
    feature_dic = {
            'weekday': map(weekday,d.created_time.values), # 7

            'source' : features.feature_to_int(d.source.values, # 9 
                                               category_dict =\
                                               s_d),

            'tag_type' : features.feature_to_int(d.tag_type.values, # 43
                                                 category_dict = t_d),

            'summary' : features.feature_to_int(d.summary.values,  #9 
                                                category_dict = summary_d),

            'description' : map(int, d.description > 0),
            'city': features.city_feature(d), #clusters #10
            'day_sixth': map(day_sixth,d.created_time.values), # 4
            'naive_nlp': map(features.naive_nlp,d.summary.values),
            'naive_nlp_description': map(features.naive_nlp,d.description.values),
            'summary_length':map(features.string_length,d.summary),
            'description_length':map(features.string_length,d.description),            
            'dense_neighbourhood':features.dense_neighbourhood(d),
            'angry_post':map(features.angry_post,d.summary.values),
            'angry_description':map(features.angry_post,d.description.values),
        }

    scalar_feature_dic = {
            'age': map(features.issue_age, d.created_time.values)
    }
    for k in feature_dic:
        d[k] = feature_dic[k]
    for k in scalar_feature_dic:
        d[k] = scalar_feature_dic[k]
    return d

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
        #pdb.set_trace()
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
    ''' Represents a model for one of {comments, views, votes}. Allows training
    and predicting using the model. '''

    def __init__(self):
        self.enc = None
        self.beast_encoder = None
        self.km = None
        #self.te_d = chicago_fix(self.te_d)

    def __make_features__(self, d):
#            'summary_bag_of_words':features.summary_bag_of_words(d)

        #for a in feature_dic.values():
        #    if not (len(a) == len(d.id.values)):
        #        pdb.set_trace()
        
        if self.beast_encoder is None:
            self.beast_encoder = F('tag_type') + F('source') + F('city') +\
                        F('naive_nlp') +F('summary_length') +\
                        F('angry_post') + F('description_length') +F('naive_nlp_description') +\
                        F('day_sixth')

#The following have been found to have poor predictive capacity:
# +F('weekday') +F('angry_description') # +F('description')# + F('summary')

            self.beast_encoder.fit(d)

        int_features = self.beast_encoder.transform(d).transpose()
        
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

            
            self.enc = sklearn.preprocessing.OneHotEncoder(
                                            n_values = self.beast_encoder.shape
                                               )
            encoded_features = self.enc.fit_transform(int_features).todense()
            print("Encoded feature shape: "+str(encoded_features.shape))


        else:

            print 'now computing the rest'
            corpus = d.summary.values
            dense_X = self.vectorizer.transform(corpus).toarray()

            encoded_features = self.enc.transform(int_features).todense()

            description_corpus = map(str,d.description.values)

            dense_X_description = self.description_vectorizer.transform(description_corpus).toarray()

            encoded_features = self.enc.transform(int_features).todense()



        print 'encoded features shape before:'
        print np.shape(encoded_features)
        encoded_features = np.concatenate((encoded_features,dense_X),axis=1)
        encoded_features = np.concatenate((encoded_features,dense_X_description),axis=1)

        print 'encoded features shape after:'
        print np.shape(encoded_features)

        scalar_features = np.array([d['age']]).transpose()
        features = np.concatenate([encoded_features,
                                   scalar_features],
                                  axis = 1)
        return features #encoded_features # no age

    def train(self, training_data, training_col):
        """
        Train the model from the training set.
        """
        #this stage is confusing, s_d needs to work for
        #__make_features__ to work properly


        #return the encoded set of features
        tr_features = self.__make_features__(training_data)

        start = time.time()
        self.regressor.fit(tr_features, tog(training_col))
        print 'Elapsed time %f' %(time.time() - start)

    def predict(self, data = None, training_set = True):
        if data is None:
            if training_set:
                data = self.tr_d
            else:
                data = self.te_d

        features = self.__make_features__(data) 
        print("features: " + str(features.shape))
        predictions = untog(self.regressor.predict(features))
        predictions = np.maximum(predictions, 0)
        return predictions

class SubmissionGenerator:
    ''' Makes a submission. Contains training and predicting code. Running the
    prediction code returns a Prediction that we can then write to a file. '''

    def __init__(self):
        self.comment_model = Model()
        self.view_model = Model()
        self.vote_model = Model()
        self.models = (self.comment_model, self.view_model, self.vote_model)

    def train(self, data):
        self.s_d = features.make_category_dict(data.source.values)
        self.t_d = features.make_category_dict(data.tag_type.values)
        self.summary_d = features.make_category_dict(data.summary.values,threshold=10)
        training_data = add_features(data, self.s_d, self.t_d, self.summary_d)

        self.vote_model.regressor =\
		ensemble.GradientBoostingRegressor(n_estimators=30,
                                                       learning_rate=0.1,
                                                       max_depth=6,
                                                       verbose=0)

        self.view_model.regressor =\
		ensemble.GradientBoostingRegressor(n_estimators=60,
                                                       learning_rate=0.1,
                                                       max_depth=6,
                                                       verbose=0)
        
        self.comment_model.regressor =\
		ensemble.RandomForestRegressor(n_estimators=30,
                                                   max_depth=4,
                                                   verbose=0)

        for (m, c) in zip(self.models,cols_to_predict):
            m.train(training_data, training_data[c].values)
        return self

    def predict(self, data):
        test_data = add_features(data, self.s_d, self.t_d, self.summary_d)
        p = []
        for m in self.models:
            p.append(m.predict(test_data))
        return Predictions(p[0], p[1], p[2])

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
        with open(file,'w') as handle:
            handle.write("id,num_views,num_votes,num_comments\n")
            for (id, comment, view, vote) in zip(d.id, self.comment_p, self.view_p, self.vote_p):
                assert(comment < 5)
                handle.write("{0},{1:.10f},{2:.10f},{3:.10f}\n".format(id, view, vote, comment))
    
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
