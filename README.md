# Various things we could try

Warning: I've written many things down here, but most of them will turn out to be relatively unimportant! Let's be careful what we spend our time on. We don't need to do all these things...

## Miscellaneous 
* Look for duplicates in test data. How common are they? (As suggested by winner of hackathon.) Interesting thing I spotted: the city often marks duplicates as closed. Doing this removes the option to vote. Can we use this?
* Move code from notebooks when it is useful to keep.
* Plot proportion of data in each small niche for the traning and test data sets.

## Mean probing
* Are the city tog means in the test data what we expect?
* How useful would it be to probe the test data set again in time?
* Can we combine the mean information we have for the remoteapi with that for the first and second halves of the test data?
* Think in terms of excess votes (>1).

## Features
* Use geographical data. Neighbourhoods of cities. Use clustering capability of scikit OR avoid the clustering and do
  some 2D histograms with numpy & matplotlib. Humans probably very good at this.
* Use locatity data: http://sedac.ciesin.columbia.edu/data/set/usgrid-summary-file3-2000-msa
* Position relative to centre of city
* Use text

## Base model
* Code machine learning data rig. (Train on 90% of the data and test on the other 10%.)
* Use SGD with more features. Get it to beat the small niche model.
* Try GBM model.
* Train machine learning on the final months of test data.
* Play around with several different models.


# Submission information
Thu, 31 Oct 2013 22:52:37 - First submission using the Beast Encoder. SGD. alpha = 0.001. pt = 0.1. shuffle = True. Pairwise encoded features: tag, source, city, weekday, description flag.
