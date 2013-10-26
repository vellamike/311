#Quick hack by MV and BD to see if correction to make togs equal
#In our submission and test data helps.

import pandas
import learn
import numpy as np
import io

source_data_path = './data/niche_means.csv'
save_to_data_path = './data/ben_corrected_validation.csv'

data = pandas.io.parsers.read_csv(source_data_path)

print data.head()
num_votes = data['num_votes'].values
num_views = data['num_views'].values
num_comments = data['num_comments'].values

prediction = (num_comments,num_views,num_votes)

#sanity check
print 'Prediction:'
print prediction

log_mean = learn.set_log_mean
scaled_prediction = log_mean(prediction)

print 'Scaled prediction:'
print scaled_prediction

id = ['id']
num_views = ['num_views']
num_votes = ['num_votes']
num_comments = ['num_comments']

ids = np.concatenate([id, data['id'].values.astype(dtype = 'S')])
comments = np.concatenate([num_comments, scaled_prediction[0]])
views = np.concatenate([num_views, scaled_prediction[1]])
votes = np.concatenate([num_votes, scaled_prediction[2]])

with open(save_to_data_path,'w') as handle:
    for (id, comment, view, vote) in zip(ids, comments, views, votes):
        handle.write("{0},{1},{2},{3}\n".format(id, view, vote, comment))
