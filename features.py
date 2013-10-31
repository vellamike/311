''' Makes and processes features. '''
import numpy as np

boundaries = [(37.4,37.7),
(37.7,37.9),
(41.24,41.3),
(41.3,41.4),
(41.6,42.1)]



def city_feature(d):
    cities = -np.ones(len(d.id))
    for (j, lat) in enumerate(d.latitude):
        for (i, bounds) in enumerate(boundaries):
            if bounds[0] < lat < bounds[1]:
                cities[j] = i
                break
        if cities[j] == -1:
            cities[j] = 4
            # Really we should figure this out.
    return cities


def make_category_dict(feature):

    category_dict = {}

    for count,string in enumerate(feature):
        if not (string in category_dict):
            category_dict[string] = count
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
