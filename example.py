"""
Loading and plotting, does not need to be run
from within interactive interpreter

field names are:
('id', 'latitude', 'longitude', 'summary', 'description', 'num_votes', 'num_comments', 'num_views', 'source', 'created_time', 'tag_type')

"""

from matplotlib import pyplot as plt
import learn
import numpy as np
import math

def plot(x,y,xlabel,ylabel):
    plt.plot(x,y,'x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


d = learn.load_data("data/train_four.csv")

chars_in_description = map(len,d.description)
chars_in_summary = map(len,d.summary)
num_votes = d.num_votes
num_views = d.num_views
num_comments = d.num_comments
latitude = d.latitude
longitude = d.longitude


def plot_multiple():
    plot(chars_in_summary,num_views,'characters in summary','number of views')
    plot(chars_in_description,num_views,'characters in description','number of views')
    plot(latitude,num_views,'latitude','number of views')
    plot(longitude,num_views,'longitude','number of views')
    
    log_num_views = map(math.log,num_views + 1)
    plot(chars_in_summary,log_num_views,'characters in summary','log + 1 of number of views')
    plot(chars_in_description,log_num_views,'characters in description','log + 1 of number of views')
    plot(latitude,log_num_views,'latitude','log + 1 of number of views')
    plot(longitude,log_num_views,'longitude','log + 1 of number of views')

def stats(x,y):
    """
    values of x become keys and stats conducted on values of y
    """

    values_dict = {}

    for i,j in zip(x,y):
        if i not in values_dict.keys():
            values_dict[i] = [j]
        else:
            values_dict[i].append(j)

    means_dict = {}
    stdev_dict = {}
    for key in values_dict.keys():
        values = values_dict[key]

        stdev = np.std(values)
        mean = np.mean(values)

        means_dict[key] = mean
        stdev_dict[key] = stdev

    return means_dict,stdev_dict

plot_multiple()

mean,sdev =  stats(chars_in_summary,num_views)
plot(mean.keys(),mean.values(),'Chars in summary','Mean number of views')
plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of views')

mean,sdev =  stats(chars_in_summary,num_comments)
plot(mean.keys(),mean.values(),'Chars in summary','Mean number of comments')
plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of comments')

mean,sdev =  stats(chars_in_summary,num_votes)
plot(mean.keys(),mean.values(),'Chars in summary','Mean number of votes')
plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of votes')
