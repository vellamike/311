This is the repository for our entry for the See Click Predict Fix Kaggle contest.

# Observations

These observations are from 11 Nov submission (Score 0.30339, pos #8)

- With gradient boosters increasing the number of estimators can be counter-productive
- CV can be misleading


# Priority things to try
- Separate training for remote_api_created, perhaps even separate training sets
- Data preformatting: this has to be critical, what can we do?
- Mean probes - when MV tested them the result was score worsening - did MV make a mistake?
- NLP - extremely naive implementation right now, how can it be improved?

# Lower priority, nice to have:
- Parallelize algos for quick evaluation
- With GBM, going from 40 to 30 estimators caused a very slight improvement, would 30 to 20 cause a more significant improvement? Wheere on the estimators/score curve is the right place?

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
* Use text - Natural language, bag of words vs something else.

## Base model
* Train a random forest regressor on the data. See how it compares to GBM. If it performs well, try submitting a blend of this with GBM. 
