''' Makes and processes features. '''
import numpy as np
from sklearn.cluster import KMeans
import time
from sklearn.feature_extraction.text import CountVectorizer

boundaries = [(37.4,37.7),
              (37.7,37.9),
              (41.24,41.3),
              (41.3,41.4),
              (41.6,42.1)]


#boundaries = [(37.4,37.7),
#              (37.7,37.9),
#              (41.24,41.4),#back to 4 cities - experiment
#              #(41.3,41.4),
#              (41.6,42.1)]
#
def string_length(string,bucket=10):
    try:
        length = len(string)
        if length > 200:
            length = 200
    except:
        length = 0
    return length // bucket
        

def angry_post(string):
    try:
        words = string.split()
    except:
        words = []
    is_angry = False
    for word in words:
        if word.isupper() and len(word) > 1:
            is_angry = True
    return int(is_angry)

def naive_nlp(string,keywords = None):
    """
    Example:
    >>> print naive_nlp('I often paint Pothole and Graffiti')
    """
    if keywords == None:
        keywords = ['Survey',
                    'Street Lights All / Out',
                    'Homeless',
                    'Tree',
                    'Street',
                    'Streets',
                    'Rat',
                    'Baiting',
                    'Traffic',
                    'Restaurant',
                    'Building',
                    'Pothole',
                    'Sanitation',
                    'Graffiti',
                    'Rodent',
                    'Completed',
                    'Complete',
                    'Trash',
                    'Light',
                    'Violation',
                    'Park',
                    'Other',
                    'Pavement'
                    'Theft',
                    'Snow',
                    'Plow',
                    'huge',
                    'Huge',
                    'Drug',
                    'Illegal Dumping',
                    'Illegal',
                    'Hydrant',
                    'drug',
                    'Abandoned',
                    'Sidewalk',
                    'sidewalk',
                    'Sidewalks',
                    'sidewalks',
                    '!',
                    '?',
                    '/',

                    ]

#    print 'keywords'
#    onehot = []
    feature = 0
    try:
        for i,keyword in enumerate(keywords): #make this lowercase and add s
            if keyword in string:
                feature = i + 1
    except:
        feature = 0
    return feature

#    decimal_encoding = sum(j<<i for i,j in enumerate(reversed(onehot)))
#    print decimal_encoding
#    return decimal_encoding



def dense_neighbourhood(d):
    """
    Better check all these values...
    """
    extra_box_0 = ((37.56,37.59),(-77.6,-77.54))
    extra_box_1 = ((37.52,37.57),(-77.54,-77.50))
    extra_box_2 = ((37.56,37.59),(-77.50,-77.45))
    extra_box_3 = ((37.57,37.59),(-77.54,-77.51))
    
    extra_box_4= ((37.74,37.80),(-122.3,-122.2))
    
    extra_box_5= ((41.25,41.27),(-72.95,-72.92))
    extra_box_6= ((41.28,41.3),(-72.95,-72.92))
    extra_box_7= ((41.28,41.3),(-72.92,-72.89))
    
    extra_box_8= ((41.32,41.34),(-72.93,-72.90))
    extra_box_9= ((41.31,41.34),(-72.90,-72.87))

    extra_boxes=[extra_box_0,
                 extra_box_1,
                 extra_box_2,
                 extra_box_3,
                 extra_box_4,
                 extra_box_6,
                 extra_box_7,
                 extra_box_8,
                 extra_box_9,]
    
    latitudes = d.latitude.values
    longitudes = d.longitude.values

    dense_places = []
    for lat,lon in zip(latitudes,longitudes):
        point = (lat,lon)
        dense_places.append(which_box(extra_boxes,point))
    #print dense_places
    return dense_places

def in_box(box,point):
    lat_extent = box[0]
    lon_extent = box[1]
    
    in_extent = lambda g,extent : extent[0] < g < extent[1]
    inside_box = lambda box,point : in_extent(point[0],box[0]) and in_extent(point[1],box[1])
        
    return inside_box(box,point)

def which_box(boxes,point):
    i = len(boxes) + 1
    for box_num,box in enumerate(boxes):
        if in_box(box,point):
            i = box_num
    
    return i

def summary_bag_of_words(training_data):
    vectorizer = CountVectorizer(min_df=1)
    corpus = training_data.summary
    X = vectorizer.fit_transform(corpus)
    return X.to_array()

def neighbourhoods(d):
    km = KMeans(n_clusters = 4, n_jobs = -1, n_init = 20, random_state = 7)
    spatial_data = d[['latitude', 'longitude']]
    clusters = km.fit_predict(spatial_data)
    print(km.score(spatial_data))
    return (km, clusters)

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


def make_category_dict(feature,threshold=0):

    frequencies = {}
    for s in feature:
        if s not in frequencies:
            frequencies[s] = (feature == s).sum()

    category_dict = {}
    count = 0
    for string in feature:
        if not (string in category_dict) and (frequencies[string] > threshold):
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
            int_feature.append(0)
    return int_feature
