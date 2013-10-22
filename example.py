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
import cost_function

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

num_points = len(d.num_views)

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

    mean,sdev =  stats(chars_in_summary,num_views)
    plot(mean.keys(),mean.values(),'Chars in summary','Mean number of views')
    plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of views')
    
    mean,sdev =  stats(chars_in_summary,num_comments)
    plot(mean.keys(),mean.values(),'Chars in summary','Mean number of comments')
    plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of comments')
    
    mean,sdev =  stats(chars_in_summary,num_votes)
    plot(mean.keys(),mean.values(),'Chars in summary','Mean number of votes')
    plot(sdev.keys(),sdev.values(),'Chars in summary','stdev in number of votes')

    
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

#plot_multiple()

def uniform_array(value,length):
    return np.ones(length)*value

def uniform_value_parameter_sweep(actual,title=None):
    errors = []
    fixed_views_value = np.arange(0.0,10.0,0.1)
    for i in fixed_views_value:
        predicted_num_views = uniform_array(i,num_points)
        error = cost_function.error_function(predicted_num_views,actual)
        errors.append(error)

    plt.plot(fixed_views_value,errors)
    plt.title = title
    plt.show()

uniform_value_parameter_sweep(d.num_views, title = 'param sweep for num_views')
uniform_value_parameter_sweep(d.num_comments, title = 'param sweep for num_comments')
uniform_value_parameter_sweep(d.num_votes, title = 'param sweep for num_votes')

mean_num_views = uniform_array(1.8,num_points)
mean_num_votes =  uniform_array(1.3,num_points) 
mean_num_comments = uniform_array(0.0,num_points)

mean_num_views_error = cost_function.error_function(mean_num_views,d.num_views)
mean_num_votes_error = cost_function.error_function(mean_num_votes,d.num_votes)
mean_num_comments_error = cost_function.error_function(mean_num_comments,d.num_comments)

print mean_num_views_error
print mean_num_votes_error
print mean_num_comments_error
print (mean_num_views_error + mean_num_votes_error + mean_num_comments_error) / 3.0
