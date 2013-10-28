import numpy as np
import io

# sklearn imports
from sklearn import svm
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor 
import sklearn.preprocessing
import pdb
from numpy import genfromtxt
import math
import time
import pandas
import city_extraction

cols_to_predict = ['num_comments', 'num_views', 'num_votes']

def load_data(path="data/train.csv"):
    return pandas.read_csv(path)

def make_features(d, source_dic = None, tag_dic = None, enc = None):
    int_features = np.zeros((len(d.id), 4))
    int_features[:,0] = feature_to_int(d.source.values, category_dict = source_dic) 
    # 9 values in training set
    int_features[:,1] = feature_to_int(d.tag_type.values, category_dict = tag_dic)
    int_features[:,2] = city_feature(d)
    int_features[:,3] = map(int, d.description > 0)
    # 43 values in training set
    if enc is None:
        enc = sklearn.preprocessing.OneHotEncoder()
        encoded_features =  enc.fit_transform(int_features).todense()
    else:
        encoded_features = enc.transform(int_features).todense()
    return (encoded_features, enc)

def train(d):
    s_d = make_category_dict(d.source.values)
    t_d = make_category_dict(d.tag_type.values)
    (features, enc) = make_features(d, source_dic = s_d, tag_dic = t_d)
    regressors = []
    start = time.time()
    for col_name in cols_to_predict:
        r = SGDRegressor(loss = "squared_loss", n_iter = 10, alpha = 0, power_t
                        = 0.1, shuffle = True)
        r.fit(features, tog(d[col_name].values))
        regressors.append(r)
    print(time.time() - start)
    # reg = ensemble.GradientBoostingRegressor(**params)
    #features = np.column_stack((d.chars_in_description[:k],
    #                            d.chars_in_summary[:k],
    #                            encoded_features[:k, :]
    #                           ))
    return (regressors, s_d, t_d, enc)

def city_feature(d):
    cities = -np.ones(len(d.id))
    for (j, lat) in enumerate(d.latitude):
        for (i, bounds) in enumerate(city_extraction.boundaries):
            if bounds[0] < lat < bounds[1]:
                cities[j] = i
                break
            if cities[j] == -1:
                cities[j] = 4
                # Really we should figure this out.
    return cities

def sgd_example():
    n_samples, n_features = 10, 5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    X = np.random.randn(n_samples, n_features)
    clf = linear_model.SGDRegressor()
    clf.fit(X, y)
    return clf

def make_category_dict(feature):
    category_dict = dict()
    count = 0
    for string in feature:
        if not (string in category_dict):
            category_dict[string] = count
            count += 1
    return category_dict

def feature_to_int(feature, category_dict = None):
    if category_dict is None:
        category_dict = make_category_dict(feature)
    int_feature = []
    for s in feature:
        if s in category_dict:
            int_feature.append(category_dict[s])
        else:
            print(s)
            int_feature.append(0)
    return int_feature

def tog(x):
    return np.log(x + 1)

def untog(x):
    return np.exp(x) - 1

def train_predict(d, outfile = "data/predictions.csv"):
    (regressors, s_d, t_d, enc) = train(d)
    training_predictions = predict(regressors, s_d, t_d, enc, training_set = True)
    # training set prediction error
    e1 = tog(d.num_comments.values) - tog(training_predictions[0])
    e2 = tog(d.num_views.values) - tog(training_predictions[1])
    e3 = tog(d.num_votes.values) - tog(training_predictions[2])
    error = rms(np.concatenate([e1,e2,e3]))
    print("Training set error: {0}".format(error))
    # predict test set
    test_predictions = predict(regressors, s_d, t_d, enc, training_set = False)
    
    id = ['id']
    num_views = ['num_views']
    num_votes = ['num_votes']
    num_comments = ['num_comments']

    ids = np.concatenate([id, d['id'].values.astype(dtype = 'S')])
    comments = np.concatenate([num_comments, test_predictions[0]])
    views = np.concatenate([num_views, test_predictions[1]])
    votes = np.concatenate([num_votes, test_predictions[2]])

    with open(outfile,'w') as handle:
        for (id, comment, view, vote) in zip(ids, comments, views, votes):
            handle.write("{0},{1},{2},{3}\n".format(id, view, vote, comment))
    
    return test_predictions

def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))

def predict(regressors, s_d, t_d, enc, training_set = False):
    if training_set:
        data = pandas.read_csv('data/train.csv')
    else:
        data = pandas.read_csv('data/test.csv')
    (test_features, enc) = make_features(data, source_dic = s_d, tag_dic = t_d,
                                  enc = enc)
    predictions = []
    for r in regressors:
        predictions.append(untog(r.predict(test_features)))
        predictions[-1]  = np.maximum(predictions[-1], 0)
    
    if not training_set:
        predictions = set_tog_means(predictions)
    return predictions

def set_tog_means(predictions):
    ''' Assume predictions is a 3 tuple of arrays: first comment predictions,
    then views, then votes.'''
    log_mean_comments = 0.02665824
    log_mean_views = 0.41264568
    log_mean_votes = 0.80850881
    means = (log_mean_comments,
             log_mean_views,
             log_mean_votes)

    scaled_predictions = []
    guesses = [] #TODO guesses does nothing? - MV
    for (m, p) in zip(means, predictions):
        scaled_predictions.append(set_tog_mean(p, m))
    return scaled_predictions

def set_tog_mean(arr, m):
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

if __name__ == "__main__":
    d = load_data()
    train(d)

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
