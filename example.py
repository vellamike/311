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

d = learn.load_data("data/train_four.csv")

chars_in_description = map(len,d.description)
num_votes = d.num_votes
num_views = d.num_views

plt.plot(chars_in_description,num_views,'x')
plt.xlabel('characters in description')
plt.ylabel('number of views')
plt.show()
