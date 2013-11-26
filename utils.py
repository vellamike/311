import numpy as np
import pandas
import csv

def tog(x):
    return np.log(x + 1)

def untog(x):
    return np.exp(x) - 1

def set_tog_mean(arr, m):
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

def predictions_corrector(prediction,
                          tog_means,
                          training_data = None,
                          correct_type = ['num_views']):
    """
    Correct predictions based on tog means in time.
    
    - Prediction is a 3-tuple num_views_test
        (num_views,num_votes,num_comments)
    or learnPredictions() object.
    
    - tog means is a list of 3-tuples:
        [(start_index,end_index,tog_mean),]
    
    - training_data is the training data path
    
    RETURN The scaled correction (untoged, ready for submission to Kaggle)
    in format (num_views,num_votes,num_comments)
    
    """
    
    if training_data != None:
        raise NotImplementedError("Function does not use training_data yet")

    if 'num_votes' in correct_type or 'num_comments' in correct_type:
       raise NotImplementedError("currently only works with correcting views")

        
    predicted_num_views = prediction[0]
    
    corrected_prediction = []
    for tog_mean in tog_means:
        i = tog_mean[0]
        j = tog_mean[1]        
        tog_mean = tog_mean[2]
        
        slice = predicted_num_views[i:j+1]
        
        scaled = set_tog_mean(slice,tog_mean)
        print 'scaled:'
        print scaled
        corrected_prediction = np.concatenate([corrected_prediction,scaled])
        
    return (corrected_prediction,prediction[1],prediction[2])

def write_submission(ids,final_num_views,prepared_votes,prepared_comments,outfile = 'submission_time_corrected.csv'):

    csv_list = np.array([ids.astype(int),
                         final_num_views,
                         prepared_votes,
                         prepared_comments])


    print ids[30]
    
    names = ['id','num_views','num_votes','num_comments']
    df = pandas.DataFrame(csv_list.T,columns=names)

    df['id']  = df['id'].apply(int)
    
    df.to_csv(outfile, index=False, header=True, sep=',')


##example:
#prediction = pandas.read_csv('/home/mike/dev/311/data/predictions.csv')
#
#ids = prediction.id.values
#predicted_num_views = prediction.num_views.values
#predicted_num_votes = prediction.num_votes.values
#predicted_num_comments = prediction.num_comments.values
#
#prediction = (predicted_num_views,
#              predicted_num_votes,
#              predicted_num_comments)
#
#tog_means = [(0,32000,0.420261689267326),
#             (32001,74999,0.426933736209156),
#             (75000,110001,0.390166529477941),
#             (110002,149575,0.410843725350216)
#             ]
#
#scaled_prediction = predictions_corrector(prediction,tog_means)
#write_submission(ids,
#                 scaled_prediction[0],
#                 scaled_prediction[1],
#                 scaled_prediction[2])
