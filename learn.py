import numpy as np
import io
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
import pdb
from numpy import genfromtxt
import math
import time

def load_data(path="data/train_four.csv"):
    """
    Output of this function appears to be a structured array,
    which does cause some headaches/benefits, depending on how you look at it:
    http://docs.scipy.org/doc/numpy/user/basics.rec.html
    """
    
    data = np.recfromcsv(path, delimiter = ";", invalid_raise = False)
    return data

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

def add_features(d):
    d.chars_in_description = map(len,d.description)
    d.chars_in_summary = map(len,d.summary)
    d.tog_num_comments = map(math.log,d.num_comments + 1)
    d.tog_num_views = map(math.log,d.num_views + 1)
    d.tog_num_votes = map(math.log,d.num_votes + 1)

def train(d):
    add_features(d)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
                        'learning_rate': 0.01, 'loss': 'ls'}
    n = len(d.tag_type)
    d.int_source = feature_to_int(d.source)
    d.int_tag_type = feature_to_int(d.tag_type)
    
    enc = sklearn.preprocessing.OneHotEncoder()
    int_features = np.zeros((n,2))
    int_features[:,0] = d.int_source # 9 values
    int_features[:,1] = d.int_tag_type # 43 values
    enc.fit(int_features)
    encoded_features = enc.transform(int_features).todense()
    # OHE -- one hot encoded 
    # initial run. Train on: chars_in_description, chars_in_summary,
    # city (OHE), source
    # (OHE), tag_type (OHE)
    # thought about using created_time, but since the test set is from a later
    # period, it will probably not be useful. 

    clf = ensemble.GradientBoostingRegressor(**params)
    #pdb.set_trace()
    k = -1
    features = np.column_stack((d.chars_in_description[:k],
                                d.chars_in_summary[:k],
                                encoded_features[:k, :]
                               ))
    make_vw_training_set(d, features)
    return
    start = time.time()
    clf.fit(features, d.tog_num_views[:k])
    print(time.time() - start)
    return clf

def make_category_dict(feature):
    category_dict = dict()
    count = 0
    for string in feature:
        if not (string in category_dict):
            category_dict[string] = count
            count += 1
    return category_dict

def feature_to_int(feature):
    category_dict = make_category_dict(feature)
    int_feature = np.array([category_dict[s] for s in feature])
    return int_feature


def predict(clf):
    predictions = set_log_mean(clf.predict(X_test))
    mse = mean_squared_error(y_test, predictions)
    print("MSE: %.4f" % mse)
    pass

def set_log_mean(predictions):
    ''' Assume predictions is a 3 tuple of arrays: first comment predictions,
    then views, then votes.'''
    log_mean_comments = 0.02665824
    log_mean_views = 0.41264568
    log_mean_votes = 0.80850881
    means = (log_mean_comments,
             log_mean_views,
             log_mean_votes)
   
    scaled_predictions = []
    guesses = []
    for (m, p) in zip(means, predictions):
        scale_factor_lb = 0
        scale_factor_ub = 2
        while scale_factor_ub - scale_factor_lb > 10**(-7):
            guess = (scale_factor_ub + scale_factor_lb) /2.0
            mean = np.mean(np.log(1+ guess * p))
            if mean > m:
                scale_factor_ub = guess
            else:
                scale_factor_lb = guess
        scaled_predictions.append(guess * p)
        guesses.append(guess)
    print('''Fudge factors: {{comments:{0}}}, {{views: {1}}}, {{votes: \
{2}}}.'''.format(guesses[0],guesses[1],guesses[2]))
    return scaled_predictions

if __name__ == "__main__":
    d = load_data()
    train(d)


