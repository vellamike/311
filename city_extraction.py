# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Dividing data into separate cities

# <codecell>

import pandas
data = pandas.read_csv('/home/mike/dev/311/data/train.csv')

# <headingcell level=3>

# Define our cities

# <codecell>

city_1_boundaries = (37.4,37.7)
city_2_boundaries = (37.7,37.9)
city_3_boundaries = (41.24,41.3)
city_4_boundaries = (41.3,41.4)
city_5_boundaries = (41.6,42.1)

# <headingcell level=3>

# Method to separate them out

# <codecell>

def select(data,city):
    """
    Return a data frame for the city within the correct latitudes
    """
    latitude_min = city[0]
    latitude_max = city[1]
    selected = data[(data.latitude >= latitude_min) & (data.latitude < latitude_max)]
    return selected

# <codecell>

city_1 = select(data,city_1_boundaries)
city_2 = select(data,city_2_boundaries)
city_3 = select(data,city_3_boundaries)
city_4 = select(data,city_4_boundaries)
city_5 = select(data,city_5_boundaries)

# <headingcell level=3>

# See an example of some of the data

# <codecell>

city_2.head()

# <headingcell level=3>

# Sanity check

# <codecell>

assert (data.id.size == city_1.id.size + city_2.id.size + city_3.id.size + city_4.id.size + city_5.id.size)

# <codecell>


