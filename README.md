# Various things we could try

Warning: I've written many things down here, but most of them will turn out to be relatively unimportant! Let's be careful what we spend our time on. We don't need to do all these things...

## Miscellaneous 
* Move code from notebooks when it is useful to keep.
* Write readme
* Plot proportion of data in each small niche for the traning and test data sets.
* ~~Add assert statement to stop us predicting that anything has less than one vote.~~
* Look for duplicates in test data. How common are they? (As suggested by winner of hackathon.)

## Mean probing
* ~~Figure out why the means have gone down~~
* ~~How do the means compare to the means of the last two months of the training data.~~
* Are the city tog means in the test data what we expect?
* How useful would it be to probe the test data set again in time?
* Can we combine the mean information we have for the remote_api with that for the first and second halves of the test data?

## Features
* Use geographical data. Neighbourhoods of cities. Use clustering capability of scikit
* Position relative to centre of city
* Use text

## Base model
* Code machine learning data rig. (Train on 90% of the data and test on the other 10%.)
* Use SGD with more features. Get it to beat the small niche model.
* Try GBM model.
* Train machine learning on the final months of test data.
* Play around with several different models.


# Submission information
Thu, 31 Oct 2013 22:52:37 - First submission using the Beast Encoder. SGD. alpha = 0.001. p_t = 0.1. shuffle = True. Pairwise encoded features: tag, source, city, weekday, description flag.
